# vllm batch

import json
import argparse
from pathlib import Path
import os
import re
import shutil
from tqdm import tqdm
from model import APIModel
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from cache import Cache
from model import APIModel
from collections import defaultdict, Counter


from datetime import datetime


def setup_logging(log_dir):
    """Per-run log file <log_dir>/evaluation_<timestamp>.log (worker processes inherit the configuration)."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, datetime.now().strftime("evaluation_%Y%m%d_%H%M%S.log")),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )



def execute_function(function, response):
    try:
        if response is None:
            return None, "Empty response"

        # prepare the execution namespace
        local_vars = {"response": response}

        # run the import statements
        import_statements = re.findall(
            r"^\s*(import\s+\S+|from\s+\S+\s+import\s+\S+)", function, re.MULTILINE
        )
        for statement in import_statements:
            exec(statement, globals())

        # define the check function
        exec(function, globals(), local_vars)

        if "check_following" in local_vars:
            result = local_vars["check_following"](local_vars["response"])
            return result, None
        else:
            return None, "check_following not defined"

    except Exception as e:
        error_msg = f"Error executing function: {e}, the function is :{function}"
        logging.error(error_msg)
        # import pdb;pdb.set_trace()

        return None, error_msg


def code_checker(function, response):
    return execute_function(function, response)


class JudgeUnavailableError(RuntimeError):
    """The judge API kept failing after all retries (auth / network / outage): abort instead of producing None scores."""


JUDGE_FAIL_EXIT_CODE = 3


def llm_checker(model, response, prompt, llm_checker_type = "llm"):
    try:
        if "{response}" not in prompt:
            prompt += "\n\nHere is model response: {response}"
        prompt = prompt.replace("{response}", response)
        
        response = model.generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        if response is None and getattr(model, "last_filtered", None):
            # rejected by the server-side content filter 3 times in a row (triggered by the model response):
            # skip this constraint (excluded from numerator and denominator), do not abort
            logging.warning(f"judge content filtered 3 times, skipped: {model.last_filtered[:200]}")
            return None
        if response is None and getattr(model, "last_error", None):
            raise JudgeUnavailableError(f"judge request failed 10 times in a row: {model.last_error[:300]}")
        if response and "</think>" in response:
            response = response.split("</think>")[1].strip()
        if model.cache.add_n > 4:
            model.cache.save_cache()
        return response
    except JudgeUnavailableError:
        raise
    except Exception as e:
        logging.error(f"LLM checker error: {e}")
        return None


def _is_prediction_null(vert):
    """No model output in the prediction stage (dropped instruction added back from the reference, or empty content)."""
    return bool(vert.get("prediction_null")) or not (((vert.get("output") or {}).get("content")) or "").strip()



def _item_key(vert):
    """Key aligning prediction outputs with the reference set (no global id: agent_name + input + constraint set)."""
    cons = [(c.get("id"), c.get("desc"), c.get("dimension"), json.dumps(c.get("type"), sort_keys=True))
            for c in vert["constraints"]]
    return json.dumps([vert.get("agent_name"), vert["input"], cons], ensure_ascii=False, sort_keys=True)


def find_missing(reference, data):
    """Reference instructions missing from the evaluation input (multiset difference, duplicates tolerated)."""
    have = Counter(_item_key(v) for v in data)
    missing = []
    for v in reference:
        k = _item_key(v)
        if have[k] > 0:
            have[k] -= 1
        else:
            missing.append(v)
    return missing


def process_item(args_tuple):
    vert = args_tuple
    content = (vert.get("output") or {}).get("content") or ""
    if not content.strip():
        # no output from the prediction stage: evaluate the empty response "" normally (code / llm / condition checks all see "")
        vert["prediction_null"] = True
        content = ""

    cache_obj = Cache(args.cache)
    model = APIModel(cache_obj, args.llm_url, args.llm_backbone, args.api_key,
                     reasoning_effort=args.reasoning_effort, timeout=args.judge_timeout)
    try:
        for j, item in enumerate(vert["constraints"]):
            if "score" in vert["constraints"][j]:
                continue

            response = content
            if "</think>" in response:
                response = response.split("</think>")[1].strip()

            # reasons for score None: judge_null / judge_unparsed (judge failure, excluded from the denominator)
            #                         condition_not_met (constraint not applicable) / code_error
            null_reason = None
            skip_llm_cond = False   # set by code_conditional_check: skip the llm_conditional_check that follows
            for e in item["evaluation"]:
                if e["type"] == "code_conditional_check":
                    # cheap code pre-check: True -> the condition holds, skip the following llm_conditional_check;
                    # False -> let the judge decide
                    hit, error = code_checker(e["exec"], response)
                    if error:
                        logging.error(f"Code conditional check error: {error}")
                        response = None
                        null_reason = "code_error"
                        break
                    skip_llm_cond = bool(hit)
                    continue

                if e["type"] == "llm_conditional_check":
                    if skip_llm_cond:
                        skip_llm_cond = False
                        continue
                    cond = llm_checker(model, response, e["exec"], llm_checker_type="llm_conditional_check")
                    if cond is None:
                        null_reason = "judge_null"
                    elif "YES" not in cond:
                        null_reason = "condition_not_met"
                    if null_reason:
                        response = None
                        break
                    continue

                if e["type"] == "llm":
                    response = llm_checker(model, response, e["exec"], llm_checker_type="llm")
                    if not response:
                        response = None
                        null_reason = "judge_null"
                        break
                elif e["type"] == "code":
                    response, error = code_checker(e["exec"], response)
                    if error:
                        logging.error(f"Code checker error: {error}")
                        response = None
                        null_reason = "code_error"
                        break
                    if response is None:
                        # a code checker returning None means "not applicable to this response"
                        # (e.g. the tool was never called); treated like an unmet condition
                        null_reason = "condition_not_met"
                        break

            if isinstance(response, str):
                vert["constraints"][j]["llm_output"] = response
                if "YES" in response:
                    response = True
                elif "NO" in response:
                    response = False
                else:
                    logging.warning(f"Unexpected response format: {response}")
                    response = None
                    null_reason = "judge_unparsed"

            vert["constraints"][j]["score"] = response
            if response is None:
                vert["constraints"][j]["null_reason"] = null_reason or "unknown"
        model.cache.save_cache()   # flush the judge cache after every instruction (otherwise small tails are lost)
        return vert

    except JudgeUnavailableError:
        raise   # judge down: let main abort the whole evaluation (exit 3)
    except Exception as e:
        logging.error(f"Processing item error: {e}")
        return vert  # keep the instruction so nothing is lost


def add_missing_from_reference(results, data, reference_file):
    """Instructions whose prediction failed are absent from eval.json: add them back from the reference set as empty responses."""
    n_dropped = 0
    if reference_file and reference_file.lower() != "none" and os.path.exists(reference_file):
        with open(reference_file, "r", encoding="utf-8") as reader:
            reference = json.load(reader)
        missing = find_missing(reference, data)
        n_dropped = len(missing)
        for vert in missing:
            vert = dict(vert)
            vert["output"] = {"content": ""}
            vert["prediction_null"] = True
            vert["constraints"] = [dict(c) for c in vert["constraints"]]
            results.append(vert)
        print(f"Reference: {reference_file} ({len(reference)} items); "
              f"dropped in prediction stage, added back as empty response: {n_dropped}")
    else:
        print("No reference file: prediction-dropped instructions are NOT counted.")
    return results


def _infer_null_reason(constraint, is_prediction_null=False):
    """Older result files have no null_reason field; infer why score is None from what is stored:
        no score field                        -> not_evaluated (the instruction aborted half-way)
        llm_output present                    -> judge_unparsed (judge answered without YES/NO)
        chain contains llm_conditional_check  -> condition_not_met (constraint not applicable)
        chain is code only                    -> code_error (old files only; new files store condition_not_met when a code checker returns None)
        otherwise (llm, no output)            -> judge_null
    """
    if "null_reason" in constraint:
        return constraint["null_reason"]
    if "score" not in constraint:
        return "prediction_null_unscored" if is_prediction_null else "not_evaluated"
    if "llm_output" in constraint:
        return "judge_unparsed"
    types = [e["type"] for e in constraint.get("evaluation", [])]
    if "llm_conditional_check" in types:
        return "condition_not_met"
    if types and all(t == "code" for t in types):
        return "code_error"
    return "judge_null"


def compute_accuracy(results):
    """Build the accuracy report (CSR / ISR / breakdowns / failure statistics) from scored results.
    Rules: instructions without model output are evaluated as the empty response "" (code checks mostly False,
           condition checks mostly not met); constraints the judge failed on, non-applicable conditional constraints
           and code errors are excluded from both numerator and denominator and reported separately."""
    total_instructions = len(results)
    total_constraints = sum(len(v["constraints"]) for v in results)
    pred_null_instr = 0
    pred_null_cons = 0
    pn_outcomes = Counter()      # outcome of the constraints of no-output instructions (pass / fail / None reasons)
    null_reasons = Counter()
    model_stats = defaultdict(lambda: {"success": 0, "total": 0})
    csr_stats = {"success": 0, "total": 0}
    isr_stats = {"success": 0, "total": 0}
    for vert in results:
        constraint_scores = []
        is_pn = _is_prediction_null(vert)
        if is_pn:
            pred_null_instr += 1
            pred_null_cons += len(vert["constraints"])
        for constraint in vert["constraints"]:
            score = constraint.get("score")
            if score is None:
                reason = _infer_null_reason(constraint, is_pn)
                null_reasons[reason] += 1
                if is_pn:
                    pn_outcomes[reason] += 1
                continue
            if is_pn:
                pn_outcomes["pass" if score is True else "fail"] += 1

            constraint_scores.append(score)

            # --- Dimension (presentation type)
            dim = constraint.get("dimension", "unknown")
            assert dim in ["unconditional", "conditional", "example_driven"]
            model_stats[dim]["total"] += 1
            if score is True:
                model_stats[dim]["success"] += 1

            # --- Type
            types = constraint["type"] if isinstance(constraint["type"], list) else [constraint["type"]]
            for t in types:
                assert t in ["formatting", "semantic", "resource"]
                model_stats[t]["total"] += 1
                if score is True:
                    model_stats[t]["success"] += 1

            # --- Meta constraints
            if constraint.get("is_meta", False):
                model_stats["meta"]["total"] += 1
                if score is True:
                    model_stats["meta"]["success"] += 1

            # --- CSR
            csr_stats["total"] += 1
            if score is True:
                csr_stats["success"] += 1

        # --- ISR
        if len(constraint_scores) > 0:
            isr_stats["total"] += 1
            if all(constraint_scores):
                isr_stats["success"] += 1

    map_constraint = {
        "unconditional": "vanilla",
        "conditional": "condition",
        "example_driven": "example",
        "formatting": "formatting",
        "semantic": "semantic",
        "resource": "tool",
    }

    def _acc(stats):
        return {
            "accuracy": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
            "correct": stats["success"],
            "total": stats["total"],
        }

    judged_constraints = total_constraints - null_reasons["prediction_null_unscored"]   # constraints that actually went through the judge
    judge_null = null_reasons["judge_null"] + null_reasons["judge_unparsed"] + null_reasons["not_evaluated"]
    accuracy_report = {
        "CSR": _acc(csr_stats),
        "ISR": _acc(isr_stats),
        "failure_stats": {
            "total_instructions": total_instructions,
            "total_constraints": total_constraints,
            "prediction_null": {            # instructions without model output, evaluated as ""
                "note": "instructions with no model output; evaluated as empty response '' "
                        "(code/llm/condition checks all see ''). constraint_outcomes = how their constraints ended up",
                "null_instructions": pred_null_instr,
                "null_instruction_rate": pred_null_instr / total_instructions if total_instructions else 0,
                "constraints_in_null_instructions": pred_null_cons,
                "constraint_outcomes": dict(pn_outcomes),
            },
            "judge_null": {                 # judge failures: no response / unparsable / aborted, excluded from both numerator and denominator
                "note": "constraints the judge failed to score; excluded from both numerator and denominator",
                "null_constraints": judge_null,
                "rate_over_judged_constraints": judge_null / judged_constraints if judged_constraints else 0,
                "no_response": null_reasons["judge_null"],
                "unparsed": null_reasons["judge_unparsed"],
                "not_evaluated": null_reasons["not_evaluated"],
            },
            "other_excluded": {             # excluded from both numerator and denominator, not judge failures
                "condition_not_met": null_reasons["condition_not_met"],
                "code_error": null_reasons["code_error"],
                "prediction_null_unscored": null_reasons["prediction_null_unscored"],
            },
        },
        "by_constraint_presentation_type": {},
        "by_constraint_type": {},
    }

    print("\n Accuracy by Presentation Type (dimension):")
    for dimension in ["unconditional", "conditional", "example_driven"]:
        r = _acc(model_stats[dimension])
        accuracy_report["by_constraint_presentation_type"][map_constraint[dimension]] = r
        print(f"  - {map_constraint[dimension]}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")

    print("\n Accuracy by Constraint Type:")
    for t in ["formatting", "semantic", "resource"]:
        r = _acc(model_stats[t])
        accuracy_report["by_constraint_type"][map_constraint[t]] = r
        print(f"  - {map_constraint[t]}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
    accuracy_report["meta_constraints"] = _acc(model_stats["meta"])
    r = accuracy_report["meta_constraints"]
    print(f"\n Meta constraints: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")

    fs = accuracy_report["failure_stats"]
    print(f"\n Prediction null (evaluated as ''): {pred_null_instr}/{total_instructions} instructions, "
          f"containing {pred_null_cons}/{total_constraints} constraints -> {dict(pn_outcomes)}")
    print(f" Judge null (excluded): {judge_null}/{judged_constraints} judged constraints "
          f"(no_response={null_reasons['judge_null']}, unparsed={null_reasons['judge_unparsed']}, "
          f"not_evaluated={null_reasons['not_evaluated']})")
    print(f" Other excluded: condition_not_met={null_reasons['condition_not_met']}, "
          f"code_error={null_reasons['code_error']}, prediction_null_unscored={null_reasons['prediction_null_unscored']}")
    print(f"\nConstraint Success Rate (CSR): {accuracy_report['CSR']['accuracy']:.4f} "
          f"({csr_stats['success']}/{csr_stats['total']})")
    print(f"Instruction Success Rate (ISR): {accuracy_report['ISR']['accuracy']:.4f} "
          f"({isr_stats['success']}/{isr_stats['total']})")

    return accuracy_report


def score_only(args):
    """Re-compute accuracy.json from an already scored results.json (adds prediction-dropped instructions). No judge calls."""
    with open(args.input_file, "r", encoding="utf-8") as reader:
        results = json.load(reader)
    results = add_missing_from_reference(results, list(results), args.reference_file)
    accuracy_report = compute_accuracy(results)
    n_unscored = accuracy_report["failure_stats"]["other_excluded"]["prediction_null_unscored"]
    if n_unscored:
        print(f"\n[score_only] {n_unscored} constraints from prediction-dropped instructions are UNSCORED (excluded). "
              f"Run WITHOUT --score_only on this results.json to score only these gaps (scored constraints are skipped).")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_file))
    os.makedirs(out_dir, exist_ok=True)
    accuracy_file = os.path.join(out_dir, "accuracy.json")
    if os.path.exists(accuracy_file):
        shutil.copy(accuracy_file, accuracy_file + ".bak")   # keep the previous report
    with open(accuracy_file, "w", encoding="utf-8") as acc_file:
        json.dump(accuracy_report, acc_file, indent=4, ensure_ascii=False)
    print(f"\n Accuracy report saved to: {accuracy_file}")


def load_partial(partial_file):
    """Load the checkpoint file results.partial.jsonl (one line per scored instruction); a truncated last line is dropped."""
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


def _abort_pool(executor, futures):
    """Judge down: cancel pending tasks and kill every worker.
    The process list must be taken before shutdown(): shutdown sets executor._processes to None and the
    workers would survive as orphans that keep calling the judge."""
    procs = list((getattr(executor, "_processes", None) or {}).values())
    for f_ in futures:
        f_.cancel()
    executor.shutdown(wait=False, cancel_futures=True)
    for p_ in procs:
        try:
            p_.kill()
        except Exception:
            pass


def main(args):
    with open(args.input_file, "r", encoding="utf-8") as reader:
        data = json.load(reader)
    data = add_missing_from_reference(data, list(data), args.reference_file)

    # output directory: <output_dir>/<input file stem>/
    file_path = args.input_file.split("/")[-1].split(".")[0]
    output_folder = Path(args.output_dir) / file_path
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = os.path.join(output_folder, "results.json")
    partial_file = os.path.join(output_folder, "results.partial.jsonl")

    # resume support: every scored instruction is appended to results.partial.jsonl; a re-run only scores the
    # remaining instructions, and results.json is written (and the checkpoint removed) once everything is done
    results = load_partial(partial_file)
    if results:
        done = Counter(_item_key(v) for v in results)
        todo = []
        for v in data:
            k = _item_key(v)
            if done[k] > 0:
                done[k] -= 1
            else:
                todo.append(v)
        print(f"[resume] {len(results)} items already scored in {partial_file}; "
              f"{len(todo)}/{len(data)} remaining", flush=True)
    else:
        todo = data

    # data = []
    # for file_path in os.listdir(args.input_file):
    #     full_path = os.path.join(args.input_file, file_path)
    #     with open(full_path, 'r', encoding='utf-8') as file:
    #         file_data = json.load(file)
    #         data.extend(file_data)

    # model = APIModel(args.llm_url, args.llm_backbone)

    with open(partial_file, "a", encoding="utf-8") as pf, \
            ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, vert) for vert in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
                pf.write(json.dumps(result, ensure_ascii=False) + "\n")   # checkpoint after every instruction
                pf.flush()
            except JudgeUnavailableError as e:
                logging.error(f"judge unavailable, aborting: {e}")
                pf.flush()
                _abort_pool(executor, futures)
                print(f"\nJUDGE_UNAVAILABLE: {e} ({len(results)}/{len(data)} instructions done, a re-run resumes from here)", flush=True)
                os._exit(JUDGE_FAIL_EXIT_CODE)
            except Exception as e:
                logging.error(f"Future error: {e}")

    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)

    print(f"Processing completed. Total processed: {len(results)}")
    print(f"Results saved to: {output_file}")
    print(f"Logs: {args.log_dir}/evaluation_*.log")

    accuracy_report = compute_accuracy(results)

    # save accuracy.json
    accuracy_file = os.path.join(output_folder, f"accuracy.json")
    with open(accuracy_file, "w", encoding="utf-8") as acc_file:
        json.dump(accuracy_report, acc_file, indent=4, ensure_ascii=False)
    # everything scored: the checkpoint file is no longer needed
    if os.path.exists(partial_file):
        os.remove(partial_file)


    print(f"\n Accuracy report saved to: {accuracy_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process constraints with parallel execution.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output directory (with --score_only defaults to the directory of input_file)")
    parser.add_argument("--score_only", action="store_true",
                        help="re-compute the statistics only: input_file is a scored results.json, no judge calls")
    parser.add_argument("--reference_file", type=str, default="data/eval.json",
                        help="full instruction set; instructions dropped in the prediction stage are scored as empty responses. `none` disables")
    parser.add_argument('--cache', type=str, default=None,
                        help="judge cache file; default .cache/<llm_backbone>.pkl (one per judge model)")
    parser.add_argument("--llm_backbone", type=str, default="gpt-5.6-luna", help="judge model name")
    parser.add_argument("--llm_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        help="judge API base URL (default: $OPENAI_BASE_URL or https://api.openai.com/v1)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="judge API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                        choices=["minimal", "low", "medium", "high", "none"],
                        help="reasoning effort for gpt-5 family judges; none = not sent (server default). Ignored by other models")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="judge temperature; 0.0 enables cache hits, other values always re-query (gpt-5.6-luna only accepts 1)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="judge output limit; sent as max_completion_tokens for gpt-5 family models")
    parser.add_argument("--judge_timeout", type=float, default=600,
                        help="per-request judge timeout in seconds; a timeout counts as a failed attempt (10 attempts, 3s/6s/9s... back-off)")

    parser.add_argument("--log_dir", type=str, default="logs", help="directory for the evaluation_<timestamp>.log file")
    args = parser.parse_args()
    setup_logging(args.log_dir)
    if args.reasoning_effort == "none":
        args.reasoning_effort = None
    if args.cache is None:
        args.cache = f".cache/{args.llm_backbone.replace('/', '_')}.pkl"

    if args.score_only:
        score_only(args)
    else:
        if not args.output_dir:
            parser.error("--output_dir is required unless --score_only")
        main(args)
