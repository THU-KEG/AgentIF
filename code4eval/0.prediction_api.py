
import json
import argparse
from pathlib import Path
import os
import re
from tqdm import tqdm
from cache import Cache
from model import APIModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time


def build_sampling(args):
    """Turn the sampling arguments into (top_p, presence_penalty, extra_body) for generate_chat.
    Parameters left at None are not sent, the server default applies."""
    extra = {}
    for k in ("top_k", "min_p", "repetition_penalty"):
        v = getattr(args, k, None)
        if v is not None:
            extra[k] = v
    if args.enable_thinking in ("true", "false"):
        extra["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking == "true"}
    if getattr(args, "extra_body", None):
        extra.update(json.loads(args.extra_body))   # vendor specific fields, e.g. {"thinking": {"type": "enabled"}}
    return args.top_p, args.presence_penalty, (extra or None)


_PREFIX_CACHE = {}


def apply_system_prefix(args, messages):
    """Prepend a vendor-required system prompt (JSON file {"role": "system", "content": <str or chunk list>}).
    If the instruction has its own system message it is appended after the prefix."""
    path = getattr(args, "system_prompt_prefix_json", None)
    if not path:
        return messages
    if path not in _PREFIX_CACHE:
        with open(path, "r", encoding="utf-8") as f:
            _PREFIX_CACHE[path] = json.load(f)
    prefix = _PREFIX_CACHE[path]
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        own = msgs.pop(0)["content"]
        own_chunks = own if isinstance(own, list) else [{"type": "text", "text": own}]
        pc = prefix["content"]
        if isinstance(pc, str):
            merged = {"role": "system", "content": pc + "\n\n" + "".join(c.get("text", "") for c in own_chunks)}
        else:
            merged = {"role": "system", "content": list(pc) + [{"type": "text", "text": "\n\n"}] + own_chunks}
        return [merged] + msgs
    return [dict(prefix)] + msgs


_HARMONY_SEG = re.compile(r"<\|channel\|>(\w+)(.*?)<\|message\|>(.*)", re.S)
_HARMONY_END = re.compile(r"(<\|end\|>|<\|call\|>|<\|return\|>)\s*$")


def parse_harmony(raw):
    """Split a raw gpt-oss Harmony transcript into (reasoning, content, channels).
    `analysis` = reasoning. `content` is the `final` channel, or the last `commentary` channel when there is no final
    (the model sent its answer as a tool call; vLLM's chat endpoint drops it when no tools are defined, so the raw
    text is kept here and left to the judge)."""
    pieces = raw.split("<|start|>assistant")
    reasoning, finals, commentaries, channels = [], [], [], []
    for piece in pieces:
        m = _HARMONY_SEG.search(piece)
        if not m:
            continue
        ch, header, text = m.group(1), m.group(2).strip(), _HARMONY_END.sub("", m.group(3))
        if ch == "commentary" and header:        # keep the recipient header (e.g. "to=functions.get_company_info <|constrain|>json")
            text = header + "\n" + text
        channels.append(ch)
        if ch == "analysis":
            reasoning.append(text)
        elif ch == "final":
            finals.append(text)
        elif ch == "commentary":
            commentaries.append(text)
    content = finals[-1] if finals else (commentaries[-1] if commentaries else "")
    return "\n".join(reasoning), content.strip(), channels


def raw_harmony_generate(args, model, messages, max_tokens, extra_body):
    """gpt-oss only: vLLM's chat endpoint always applies Harmony parsing for gpt_oss and has no switch.
    Use /tokenize (chat template, supports chat_template_kwargs.reasoning_effort) + /v1/completions
    (skip_special_tokens=false) and parse the channels ourselves."""
    base = args.llm_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    extra = dict(extra_body or {})
    ctk = dict(extra.pop("chat_template_kwargs", {}) or {})
    if "reasoning_effort" in extra:
        ctk["reasoning_effort"] = extra.pop("reasoning_effort")
    tok = requests.post(f"{base}/tokenize", json={"model": args.llm_backbone, "messages": messages,
                                                 "add_generation_prompt": True, "chat_template_kwargs": ctk}, timeout=120)
    tok.raise_for_status()
    ids = tok.json()["tokens"]
    if args.max_model_len:
        max_tokens = max(1, min(max_tokens, args.max_model_len - len(ids) - 16))
    payload = {"model": args.llm_backbone, "prompt": ids, "max_tokens": max_tokens, "temperature": args.temperature,
               "skip_special_tokens": False}
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    payload.update(extra)   # top_k / min_p / repetition_penalty are accepted by vLLM's completions endpoint directly
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(f"{base}/v1/completions", json=payload, timeout=3000)
            r.raise_for_status()
            ch = r.json()["choices"][0]
            reasoning, content, channels = parse_harmony(ch["text"])
            model.last_reasoning, model.last_finish_reason, model.last_max_tokens, model.last_error = \
                reasoning, ch.get("finish_reason"), max_tokens, None
            model.last_channels = channels
            return content
        except Exception as e:
            last_err = str(e)
            print(f"raw harmony attempt {attempt + 1} failed: {last_err[:160]}")
            time.sleep(2 * (attempt + 1))
    model.last_error = last_err
    return None


def count_prompt_tokens(args, messages, extra_body):
    """Exact prompt length (after the chat template) through vLLM's /tokenize endpoint; None if unavailable."""
    base = args.llm_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    payload = {"model": args.llm_backbone, "messages": messages, "add_generation_prompt": True}
    ctk = (extra_body or {}).get("chat_template_kwargs")
    if ctk:
        payload["chat_template_kwargs"] = ctk
    try:
        r = requests.post(f"{base}/tokenize", json=payload, timeout=60)
        r.raise_for_status()
        return int(r.json()["count"])
    except Exception as e:
        print(f"/tokenize failed, fall back to server-side error retry: {str(e)[:120]}")
        return None

def process_single_entry(vert, args):
    cache_obj = Cache(args.cache)
    model = APIModel(cache_obj, args.llm_url, args.llm_backbone, api_key=args.api_key)
    input_msgs = vert["input"]
    
    for item in input_msgs:
        assert item["role"] == "user" or item["role"] == "system" or item["role"] == "assistant", print(vert["input"])
        if item["role"] == "developer":
            import pdb; pdb.set_trace()
            
    assert input_msgs[-1]["role"] == "user", print(vert["input"])
    assert input_msgs[-1]["content"] != "", print(vert["input"])

    top_p, presence_penalty, extra_body = build_sampling(args)
    input_msgs = apply_system_prefix(args, input_msgs)

    # trim max_tokens so that prompt + max_tokens fits into max-model-len
    max_tokens = args.max_tokens
    if args.max_model_len:
        n_prompt = count_prompt_tokens(args, input_msgs, extra_body)
        if n_prompt is not None:
            allowed = args.max_model_len - n_prompt - 16
            if allowed < max_tokens:
                print(f"prompt={n_prompt} tokens, shrink max_tokens {max_tokens} -> {allowed}")
                max_tokens = max(allowed, 1)

    model.last_channels = None
    if args.raw_harmony:
        response = raw_harmony_generate(args, model, input_msgs, max_tokens, extra_body)
    else:
        response = model.generate_chat(
            input_msgs,
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
            use_cache=not args.no_cache,
        )
    if response is None and getattr(model, "last_error", None):  # API failed after all retries -> this instruction fails; main() counts consecutive failures
        print(f"api failed after retries: {model.last_error[:200]}")
        return {"__failed__": model.last_error}
    if not response:      # API succeeded but no final answer (e.g. the thinking budget was exhausted) -> keep as an empty response
        print(f"empty response (finish_reason={model.last_finish_reason}, reasoning_len={len(model.last_reasoning or '')})")
        response = ""

    vert["output"] = {
        "content": response,
        "model_name": args.llm_backbone,
        "original_response": response,
        "reasoning_content": model.last_reasoning,
        "finish_reason": model.last_finish_reason,
        "max_tokens_used": model.last_max_tokens,
        "harmony_channels": getattr(model, "last_channels", None),
    }

    # import pdb; pdb.set_trace()
    if "</think>" in response and "glm" not in args.llm_backbone:
        response = response.split("</think>")[1].strip()

    vert["output"]["content"] = response.strip()
    
    model.cache.save_cache()
    return vert

PREDICTION_FAIL_EXIT_CODE = 3
FATAL_API_ERROR = re.compile(r"401|402|403|insufficient|balance|quota|unauthorized|invalid.{0,20}(api.?key|token)|billing|arrears", re.I)


def _item_key(vert):
    """Same key as 1.evaluation_api._item_key (agent_name + input + constraint set); aligns checkpointed items on resume."""
    cons = [(c.get("id"), c.get("desc"), c.get("dimension"), json.dumps(c.get("type"), sort_keys=True))
            for c in vert["constraints"]]
    return json.dumps([vert.get("agent_name"), vert["input"], cons], ensure_ascii=False, sort_keys=True)


def load_partial(partial_file):
    """Load the checkpoint file eval.partial.jsonl (one line per predicted instruction); a truncated last line is dropped."""
    results = []
    if not os.path.exists(partial_file):
        return results
    with open(partial_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[resume] {partial_file} line {i + 1} is truncated, dropped", flush=True)
    return results


def main(args):

    for file_path in os.listdir(args.input_dir):
        if file_path != "eval.json":
            continue
        full_path = os.path.join(args.input_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            print("Processing:", full_path)
            data = json.load(file)

        output_folder = Path(args.output_dir)
        output_folder.mkdir(exist_ok=True, parents=True)
        output_path = os.path.join(output_folder, file_path)
        partial_file = os.path.join(output_folder, "eval.partial.jsonl")

        # resume support: every finished instruction is appended to eval.partial.jsonl; a re-run only predicts the
        # remaining instructions, and eval.json is written (and the checkpoint removed) once everything is done
        results = load_partial(partial_file)
        if results:
            done = {}
            for v in results:
                k = _item_key(v); done[k] = done.get(k, 0) + 1
            todo = []
            for v in data:
                k = _item_key(v)
                if done.get(k, 0) > 0:
                    done[k] -= 1
                else:
                    todo.append(v)
            print(f"[resume] {len(results)} items already predicted in {partial_file}; {len(todo)}/{len(data)} remaining", flush=True)
        else:
            todo = data

        n_failed = 0
        consecutive_failed = 0
        with open(partial_file, "a", encoding="utf-8") as pf, \
                ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_single_entry, vert, args) for vert in todo]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result is None:
                    n_failed += 1
                    continue
                if "__failed__" in result:
                    # the API failed after 3 retries: abort (exit 3) when the failures look like an auth / billing /
                    # quota problem or when too many instructions fail in a row; finished items are on disk, a re-run resumes
                    n_failed += 1
                    consecutive_failed += 1
                    err = result["__failed__"] or ""
                    fatal = FATAL_API_ERROR.search(err)
                    if fatal or consecutive_failed >= args.max_consecutive_failures:
                        pf.flush()
                        reason = f"fatal api error: {err[:200]}" if fatal else f"{consecutive_failed} consecutive api failures, last: {err[:200]}"
                        print(f"\nPREDICTION_ABORTED: {reason} ({len(results)}/{len(data)} instructions done, "
                              f"saved in {partial_file}; a re-run resumes from here)", flush=True)
                        for f_ in futures:
                            f_.cancel()
                        os._exit(PREDICTION_FAIL_EXIT_CODE)
                    continue
                consecutive_failed = 0
                results.append(result)
                pf.write(json.dumps(result, ensure_ascii=False) + "\n")   # checkpoint after every instruction
                pf.flush()

        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(results, out_file, indent=4, ensure_ascii=False)

        # record every sampling / mode parameter of this run for reproducibility
        top_p, presence_penalty, extra_body = build_sampling(args)
        run_config = {
            "llm_backbone": args.llm_backbone,
            "llm_url": args.llm_url,
            "temperature": args.temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "extra_body": extra_body,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "enable_thinking": args.enable_thinking,
            "no_cache": args.no_cache,
            "system_prompt_prefix_json": args.system_prompt_prefix_json,
            "raw_harmony": args.raw_harmony,
            "num_total": len(data),
            "num_success": len(results),
            "num_empty_response": sum(1 for r in results if not (r["output"]["content"] or "").strip()),
        }
        with open(os.path.join(output_folder, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4, ensure_ascii=False)
        if os.path.exists(partial_file):
            os.remove(partial_file)   # everything predicted: the checkpoint file is no longer needed

        n_empty = run_config["num_empty_response"]
        print("Successfully process:", len(results), "| empty responses:", n_empty, "| api failed:", len(data) - len(results))

    print("total_number:", len(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multithreaded Conditional Checker")
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="results/prediction/model")
    parser.add_argument("--llm_backbone", type=str, default="", help="`model` field of the request (served model name)")
    parser.add_argument("--llm_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument('--cache', type=str, default='.cache/prediction.pkl')
    parser.add_argument('--max_tokens', type=int, default=10240)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    # ---- sampling / mode parameters (defaults: temperature=0, nothing else sent) ----
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None, help="vLLM extra_body.top_k")
    parser.add_argument("--min_p", type=float, default=None, help="vLLM extra_body.min_p")
    parser.add_argument("--presence_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None, help="vLLM extra_body.repetition_penalty")
    parser.add_argument("--enable_thinking", type=str, default="none", choices=["none", "true", "false"],
                        help="thinking switch through chat_template_kwargs.enable_thinking; none = not sent (template default)")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="max-model-len of the vLLM server; when given, max_tokens is trimmed to the remaining context")
    parser.add_argument("--no_cache", action="store_true", help="do not read or write the response cache (independent repeated runs)")
    parser.add_argument("--max_consecutive_failures", type=int, default=8,
                        help="abort the run (exit 3) after this many consecutive instructions failed all 3 API retries; "
                             "auth / billing / quota errors abort immediately")
    parser.add_argument("--system_prompt_prefix_json", type=str, default=None,
                        help="JSON file ({role: system, content: ...}) prepended to every instruction (vendor-required system prompt)")
    parser.add_argument("--extra_body", type=str, default=None,
                        help='JSON string merged into the request extra_body, e.g. \'{"thinking": {"type": "enabled"}}\'')
    parser.add_argument("--raw_harmony", action="store_true",
                        help="gpt-oss only: call /tokenize + /v1/completions and parse the Harmony channels here "
                             "(vLLM's chat endpoint drops answers sent on the commentary channel)")
    args = parser.parse_args()

    main(args)
