#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgentIF evaluation orchestrator (config driven).

For every model YAML in configs/:
    [backend=vllm]  start vLLM -> wait until healthy -> prediction for every (mode x repeat) -> stop vLLM -> evaluation
    [backend=api ]  prediction through an OpenAI-compatible endpoint -> evaluation
Models run in parallel when the GPU pool allows it. Finished jobs (an existing accuracy.json) are skipped, so the
command can simply be re-run after an interruption.

The judge is described by judge_configs/<name>.yaml (default: `judge` in configs/global.yaml, override with --judge).

Usage (from the repository root, or via `bash run.sh ...`):
    python code4eval/batch_eval.py                              # every enabled model, prediction + evaluation
    python code4eval/batch_eval.py --models Qwen3.5-4B ...      # only these models (file stem or model_name)
    python code4eval/batch_eval.py --stage prediction           # prediction only
    python code4eval/batch_eval.py --stage evaluation           # evaluation only (no vLLM, scores existing predictions)
    python code4eval/batch_eval.py --judge gpt-5.6-luna         # pick a judge config
    python code4eval/batch_eval.py --repeats 1 --input-dir data/smoke --results-root results_smoke   # quick smoke test
    python code4eval/batch_eval.py --dry-run                    # print the commands only
    python code4eval/batch_eval.py --list                       # list model and judge configs
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
import yaml

AGENTIF_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = AGENTIF_DIR / "configs"
JUDGE_DIR = AGENTIF_DIR / "judge_configs"

_print_lock = threading.Lock()
JUDGE_DOWN = threading.Event()      # set once an evaluation job exits with rc=3 (judge unreachable); pending jobs are skipped
JUDGE_FAIL_EXIT_CODE = 3


def log(tag, msg):
    with _print_lock:
        print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] [{tag}] {msg}", flush=True)


# ----------------------------------------------------------------------------- config
def expand_env(obj):
    """Expand ${VAR} / $VAR in every string of a (nested) YAML object."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    return obj


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return expand_env(yaml.safe_load(f) or {})


def resolve_env_value(d, key, default=""):
    """Value lookup used for secrets / endpoints: the environment variable named by `<key>_env` wins,
    then the literal `<key>` in the YAML, then `default`."""
    env = d.get(f"{key}_env")
    if env and os.environ.get(env):
        return os.environ[env]
    return d.get(key) or default


def list_yaml(dirpath):
    return [p for p in sorted(glob.glob(str(dirpath / "*.yaml"))) if not Path(p).name.endswith(".example.yaml")]


def load_judge(name):
    """`name` may be a judge config stem (gpt-5.6-luna), a file name (gpt-5.6-luna.yaml) or any path."""
    cands = [Path(name), AGENTIF_DIR / name, JUDGE_DIR / name, JUDGE_DIR / f"{name}.yaml"]
    path = next((c for c in cands if c.is_file()), JUDGE_DIR / f"{Path(name).stem}.yaml")
    name = Path(name).stem
    if not path.exists():
        sys.exit(f"judge config not found: {path}; available: {[Path(p).stem for p in list_yaml(JUDGE_DIR)]}")
    j = load_yaml(path)
    j.setdefault("name", name)
    j.setdefault("backbone", j["name"])
    j["base_url"] = resolve_env_value(j, "base_url", default="https://api.openai.com/v1")
    j["api_key"] = resolve_env_value(j, "api_key")
    j.setdefault("temperature", 1.0)
    j.setdefault("max_tokens", 8192)
    j.setdefault("reasoning_effort", "none")
    j.setdefault("num_workers", 8)
    j.setdefault("max_concurrent_jobs", 6)
    j["_config_path"] = str(path)
    return j


def load_models(model_filter):
    models = []
    for p in list_yaml(CONFIG_DIR):
        if Path(p).name == "global.yaml":
            continue
        c = load_yaml(p)
        c["_config_path"] = p
        c.setdefault("model_name", Path(p).stem)
        c.setdefault("backend", "vllm")
        if c["backend"] not in ("vllm", "api"):
            sys.exit(f"{p}: backend must be vllm or api, got {c['backend']}")
        explicitly_selected = bool(model_filter) and (c["model_name"] in model_filter or Path(p).stem in model_filter)
        if model_filter and not explicitly_selected:
            continue
        if not c.get("enabled", True) and not explicitly_selected:
            log("config", f"skip {c['model_name']} ({Path(p).name}): enabled=false")
            continue
        if c["backend"] == "api":
            a = c.get("api") or {}
            c["api"] = a
            a["base_url"] = resolve_env_value(a, "base_url")
            a["api_key"] = resolve_env_value(a, "api_key", default="EMPTY")
            a.setdefault("model", c["model_name"])
            c.setdefault("served_model_name", a["model"])
        else:
            c.setdefault("served_model_name", c["model_name"])
        models.append(c)
    if model_filter:
        found = {c["model_name"] for c in models} | {Path(c["_config_path"]).stem for c in models}
        missing = [m for m in model_filter if m not in found]
        if missing:
            sys.exit(f"model config not found: {missing}; available: "
                     f"{[Path(p).stem for p in list_yaml(CONFIG_DIR) if Path(p).name != 'global.yaml']}")
    return models


def resolve_environment(g):
    """Fill machine-specific defaults of global.yaml: python interpreter, vllm binary, GPU pool."""
    g["eval_python"] = g.get("eval_python") or sys.executable
    vllm_bin = g.get("vllm_bin") or ""
    if not vllm_bin:
        vllm_bin = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
    g["vllm_bin"] = vllm_bin
    pool = str(g.get("gpu_pool", "auto")).strip()
    if pool in ("", "auto"):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            pool = visible
        else:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=30).stdout
                pool = ",".join(l.strip() for l in out.strip().splitlines() if l.strip())
            except Exception:
                pool = ""
    g["gpu_pool"] = pool
    return g


# ----------------------------------------------------------------------------- GPU / port pool
class GPUPool:
    """Hand out GPUs on demand; a request blocks until `n` GPUs are free."""

    def __init__(self, gpus):
        self.all = list(gpus)
        self.free = list(gpus)
        self.cv = threading.Condition()

    def acquire(self, n, fixed=None, tag=""):
        with self.cv:
            if fixed is not None:
                want = [g.strip() for g in str(fixed).split(",") if g.strip()]
                while not all(g in self.free for g in want):
                    log(tag, f"waiting for fixed GPUs {want} (free: {self.free})")
                    self.cv.wait()
                for g in want:
                    self.free.remove(g)
                return want
            if n > len(self.all):
                log(tag, f"num_gpus={n} exceeds the pool size {len(self.all)}, using {len(self.all)}")
                n = len(self.all)
            while len(self.free) < n:
                log(tag, f"waiting for {n} GPU(s) (free: {self.free})")
                self.cv.wait()
            got, self.free = self.free[:n], self.free[n:]
            return got

    def release(self, gpus):
        with self.cv:
            self.free.extend(gpus)
            self.free.sort(key=int)
            self.cv.notify_all()


class PortPool:
    def __init__(self, base):
        self.next = int(base)
        self.lock = threading.Lock()

    @staticmethod
    def _free(port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    def acquire(self):
        with self.lock:
            while not self._free(self.next):
                self.next += 1
            port = self.next
            self.next += 1
            return port


# ----------------------------------------------------------------------------- backends
class APIBackend:
    """backend=api: nothing to deploy, use base_url / key from the config."""

    def __init__(self, c):
        self.c = c
        self.base_url = c["api"]["base_url"].rstrip("/")
        self.api_key = c["api"]["api_key"]
        self.max_model_len = c["api"].get("max_model_len")

    def start(self): pass
    def wait_ready(self): pass
    def stop(self): pass

    def smoke_test(self, dry_run):
        if dry_run:
            return
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=300)
        r = client.chat.completions.create(model=self.c["served_model_name"],
                                           messages=[{"role": "user", "content": "Say hi in one word."}], max_tokens=64)
        log(self.c["model_name"], f"api smoke test ok: {r.choices[0].message.content!r:.80}")


class VLLMServer:
    """backend=vllm: start / health check / stop a vLLM server."""

    def __init__(self, g, c, log_dir, dry_run=False, gpus=None, port=None):
        self.g, self.c = g, c
        self.name = c["model_name"]
        self.gpus = gpus if gpus is not None else str(c.get("gpus", ""))
        self.port = int(port if port is not None else c["port"])
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.api_key = "EMPTY"
        self.max_model_len = c["max_model_len"]
        self.log_path = log_dir / f"{self.name}_vllm.log"
        self.proc = None
        self.dry_run = dry_run

    def build_cmd(self):
        c = self.c
        cmd = [self.g["vllm_bin"], "serve", c["model_path"],
               "--served-model-name", c["served_model_name"],
               "--host", "0.0.0.0", "--port", str(self.port),
               "--tensor-parallel-size", str(c.get("tensor_parallel_size", 1)),
               "--max-model-len", str(c["max_model_len"])]
        tp = int(c.get("tensor_parallel_size", 1))
        n_gpu = len([x for x in str(self.gpus).split(",") if x.strip()])
        dp = int(c.get("data_parallel_size", max(1, n_gpu // tp)))   # default: one replica per tp group, use every GPU we got
        if dp > 1:
            cmd += ["--data-parallel-size", str(dp)]
        for a in c.get("vllm_args", []):
            cmd += shlex.split(str(a))
        return cmd

    def start(self):
        cmd = self.build_cmd()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpus))
        env["PATH"] = f"{Path(self.g['vllm_bin']).parent}:{env.get('PATH', '')}"
        log(self.name, f"starting vLLM (GPUs={self.gpus}, port={self.port}):\n    " + " ".join(shlex.quote(x) for x in cmd))
        if self.dry_run:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdout=open(self.log_path, "ab"), stderr=subprocess.STDOUT,
                                     env=env, cwd=str(AGENTIF_DIR), start_new_session=True)

    def wait_ready(self):
        if self.dry_run:
            return
        timeout = int(self.g["serve"].get("health_timeout_sec", 2400))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early (exit={self.proc.returncode}), see {self.log_path}")
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
                if r.status_code == 200:
                    m = requests.get(f"{self.base_url}/models", timeout=10).json()
                    names = [x["id"] for x in m.get("data", [])]
                    if self.c["served_model_name"] in names:
                        log(self.name, f"vLLM ready after {time.time() - t0:.0f}s, models={names}")
                        return
            except Exception:
                pass
            time.sleep(10)
        raise RuntimeError(f"vLLM not ready within {timeout}s, see {self.log_path}")

    def smoke_test(self, dry_run):
        """One minimal request to make sure the chat endpoint and the reasoning parser work."""
        if dry_run:
            return
        payload = {"model": self.c["served_model_name"],
                   "messages": [{"role": "user", "content": "Say hi in one word."}],
                   "max_tokens": 64, "temperature": 0}
        r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        log(self.name, f"smoke test ok: content={msg.get('content')!r:.80} "
                       f"reasoning={'yes' if msg.get('reasoning_content') or msg.get('reasoning') else 'no'}")
        # AgentIF prompts dictate a custom textual tool-call format; make sure no tool-call parser swallows it.
        payload = {"model": self.c["served_model_name"],
                   "messages": [{"role": "user", "content": 'Reply with exactly this line and nothing else: '
                                 '<tool>{"name": "get_weather", "args": {"city": "Paris"}}</tool>'}],
                   "max_tokens": 128, "temperature": 0}
        if any("enable_thinking" in mc for mc in self.c.get("modes", {}).values()):
            payload["chat_template_kwargs"] = {"enable_thinking": False}   # only when the template supports it
        r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        ok = "<tool>" in content and "get_weather" in content and not msg.get("tool_calls")
        log(self.name, f"custom tool-format check {'ok' if ok else 'WARNING'}: content={content!r:.120} "
                       f"tool_calls={msg.get('tool_calls')}")

    def stop(self):
        if self.dry_run or self.proc is None:
            return
        grace = int(self.g["serve"].get("shutdown_grace_sec", 60))
        log(self.name, "stopping vLLM ...")
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        t0 = time.time()
        while self.proc.poll() is None and time.time() - t0 < grace:
            time.sleep(2)
        if self.proc.poll() is None:
            log(self.name, "SIGTERM timed out, sending SIGKILL")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=30)
        for _ in range(30):          # wait until the GPU memory is really released
            if not self._gpu_busy() and not (busy_gpus() & {g.strip() for g in str(self.gpus).split(",")}):
                break
            time.sleep(5)
        time.sleep(5)
        log(self.name, f"vLLM stopped (exit={self.proc.returncode}), GPU memory released")

    def _gpu_busy(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout
            uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=30).stdout
            idx2uuid = {l.split(",")[0].strip(): l.split(",")[1].strip() for l in uuids.strip().splitlines() if l.strip()}
            mine = {idx2uuid[i.strip()] for i in str(self.gpus).split(",") if i.strip() in idx2uuid}
            return any(l.split(",")[0].strip() in mine for l in out.strip().splitlines() if l.strip())
        except Exception:
            return False


def make_backend(g, c, log_dir, dry_run, gpus=None, port=None):
    return VLLMServer(g, c, log_dir, dry_run, gpus=gpus, port=port) if c["backend"] == "vllm" else APIBackend(c)


# ----------------------------------------------------------------------------- jobs
def job_paths(g, judge, c, mode, run):
    root = AGENTIF_DIR / g["results_root"]
    tag = f"{c['model_name']}__{mode}"
    pred_dir = root / "prediction" / tag / f"run{run}"
    score_dir = root / "scores" / tag / f"run{run}_{judge['name']}"
    cache_dir = AGENTIF_DIR / g["cache_dir"]
    return dict(
        pred_dir=pred_dir,
        pred_file=pred_dir / "eval.json",
        score_dir=score_dir,
        accuracy=score_dir / "eval" / "accuracy.json",
        pred_cache=cache_dir / f"pred_{tag}_run{run}.pkl",
        judge_cache=cache_dir / f"judge_{judge['name']}_{tag}_run{run}.pkl",
    )


def prediction_cmd(g, c, mcfg, backend, paths):
    cmd = [g["eval_python"], str(CODE_DIR / "0.prediction_api.py"),
           "--input_dir", g["input_dir"],
           "--output_dir", str(paths["pred_dir"]),
           "--llm_backbone", c["served_model_name"],
           "--llm_url", backend.base_url,
           "--api_key", backend.api_key,
           "--cache", str(paths["pred_cache"]),
           "--no_cache",
           "--num_workers", str(c.get("prediction_num_workers", g["prediction"]["num_workers"])),   # per-model override (lower it for rate-limited APIs)
           "--max_tokens", str(mcfg["max_tokens"]),
           "--temperature", str(mcfg["temperature"])]
    if backend.max_model_len:
        cmd += ["--max_model_len", str(backend.max_model_len)]
    if "enable_thinking" in mcfg and mcfg["enable_thinking"] is not None:
        cmd += ["--enable_thinking", "true" if mcfg["enable_thinking"] else "false"]
    for k in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
        if mcfg.get(k) is not None:
            cmd += [f"--{k}", str(mcfg[k])]
    if mcfg.get("extra_body"):
        cmd += ["--extra_body", json.dumps(mcfg["extra_body"], ensure_ascii=False)]
    if c.get("request_mode") == "raw_harmony":
        cmd += ["--raw_harmony"]
    prefix = mcfg.get("system_prompt_prefix_json", c.get("system_prompt_prefix_json"))   # mode level wins (e.g. "/no_think")
    if prefix:
        cmd += ["--system_prompt_prefix_json", str(prefix)]
    return cmd


def evaluation_cmd(g, judge, paths):
    return [g["eval_python"], str(CODE_DIR / "1.evaluation_api.py"),
            "--input_file", str(paths["pred_file"]),
            "--output_dir", str(paths["score_dir"]),
            # the full instruction set: instructions dropped during prediction are scored as empty responses
            "--reference_file", str(Path(g["input_dir"]) / "eval.json"),
            "--llm_backbone", judge["backbone"],
            "--llm_url", judge["base_url"],
            "--api_key", judge["api_key"],
            "--cache", str(paths["judge_cache"]),
            "--temperature", str(judge["temperature"]),
            "--max_tokens", str(judge["max_tokens"]),
            "--reasoning_effort", str(judge["reasoning_effort"]),
            "--num_workers", str(judge["num_workers"]),
            "--log_dir", str(AGENTIF_DIR / g["log_dir"])]


def redact(cmd):
    """Hide API keys when printing commands."""
    out = list(cmd)
    for i, x in enumerate(out):
        if x == "--api_key" and i + 1 < len(out) and out[i + 1] not in ("EMPTY", ""):
            out[i + 1] = out[i + 1][:4] + "****"
    return " ".join(shlex.quote(x) for x in out)


def run_cmd(tag, cmd, log_file, dry_run):
    log(tag, "$ " + redact(cmd))
    if dry_run:
        return 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as f:
        f.write(f"\n===== {datetime.now()} =====\n".encode())
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(AGENTIF_DIR))
    return p.returncode


def prediction_complete(paths):
    """run_config.json is written only after the whole instruction set was processed, so its presence means done.
    Instructions whose API call failed after all retries (num_success < num_total) count as failures; no re-run."""
    rc = paths["pred_dir"] / "run_config.json"
    if not (paths["pred_file"].exists() and rc.exists()):
        return False
    try:
        d = json.load(open(rc))
        if d.get("num_success", 0) <= 0:
            return False
        if d.get("num_success") != d.get("num_total"):
            log("resume", f"{paths['pred_dir']}: {d.get('num_success')}/{d.get('num_total')} succeeded, "
                          f"the rest are scored as empty responses")
        return True
    except Exception:
        return False


def run_prediction_job(g, judge, c, mode, mcfg, run, backend, args, log_dir):
    """Stage 1: prediction for one (mode, run). Returns the status dict."""
    tag = f"{c['model_name']}/{mode}/run{run}"
    paths = job_paths(g, judge, c, mode, run)
    status = {"mode": mode, "run": run, "prediction": None, "evaluation": None}
    try:
        if paths["accuracy"].exists():
            log(tag, f"{paths['accuracy']} exists, skip")
            status["prediction"] = status["evaluation"] = "skipped(exists)"
        elif prediction_complete(paths):
            log(tag, "prediction already complete, skip")
            status["prediction"] = "skipped(exists)"
        elif args.stage == "evaluation":
            log(tag, "stage=evaluation but prediction is missing, skip")
            status["prediction"] = "missing"
            status["evaluation"] = "skipped(no prediction)"
        else:
            rc = run_cmd(tag, prediction_cmd(g, c, mcfg, backend, paths),
                         log_dir / f"{c['model_name']}__{mode}_run{run}_prediction.log", args.dry_run)
            status["prediction"] = "ok" if rc == 0 else f"failed(rc={rc})"
    except Exception as e:
        status["error"] = f"{e}\n{traceback.format_exc()}"
        log(tag, f"prediction error: {e}")
    return status


def run_evaluation_job(g, judge, c, status, eval_sem, args, log_dir):
    """Stage 2: evaluation for one (mode, run); vLLM is already down, only the judge API is used."""
    mode, run = status["mode"], status["run"]
    tag = f"{c['model_name']}/{mode}/run{run}"
    paths = job_paths(g, judge, c, mode, run)
    if status["evaluation"] is not None:          # already skipped / exists
        return status
    if status["prediction"] not in ("ok", "skipped(exists)"):
        status["evaluation"] = "skipped(prediction failed)"
        return status
    if args.stage == "prediction":
        status["evaluation"] = "skipped(stage=prediction)"
        return status
    if JUDGE_DOWN.is_set():
        status["evaluation"] = "skipped(judge down)"
        return status
    try:
        with eval_sem:
            if JUDGE_DOWN.is_set():
                status["evaluation"] = "skipped(judge down)"
                return status
            rc = run_cmd(tag, evaluation_cmd(g, judge, paths),
                         log_dir / f"{c['model_name']}__{mode}_run{run}_evaluation_{judge['name']}.log", args.dry_run)
        if rc == JUDGE_FAIL_EXIT_CODE:
            JUDGE_DOWN.set()
            log(tag, "judge unreachable after retries; remaining evaluation jobs are skipped "
                     "(predictions are kept, re-run with --stage evaluation once the judge is back)")
            status["evaluation"] = "failed(judge down)"
            return status
        status["evaluation"] = "ok" if rc == 0 else f"failed(rc={rc})"
        if rc == 0 and not args.dry_run and paths["accuracy"].exists():
            acc = json.load(open(paths["accuracy"]))
            status["CSR"] = acc.get("CSR", {}).get("accuracy")
            status["ISR"] = acc.get("ISR", {}).get("accuracy")
            log(tag, f"done: CSR={status['CSR']:.4f} ISR={status['ISR']:.4f}")
    except Exception as e:
        status["error"] = f"{e}\n{traceback.format_exc()}"
        log(tag, f"evaluation error: {e}")
    return status


# ----------------------------------------------------------------------------- per model
def vllm_version(g):
    try:
        py = str(Path(g["vllm_bin"]).parent / "python")
        if not Path(py).exists():
            py = g["eval_python"]
        return subprocess.run([py, "-c", "import vllm;print(vllm.__version__)"],
                              capture_output=True, text=True, timeout=120).stdout.strip().splitlines()[-1]
    except Exception:
        return "unknown"


def strip_secrets(d):
    return {k: ("****" if k == "api_key" and v else v) for k, v in d.items()} if isinstance(d, dict) else d


def run_model(g, judge, c, args, eval_sem, log_dir, pools):
    name = c["model_name"]
    t_start = datetime.now()
    meta_dir = AGENTIF_DIR / g["results_root"] / "run_metadata" / name
    meta_dir.mkdir(parents=True, exist_ok=True)
    c_meta = dict(c)
    if "api" in c_meta:
        c_meta["api"] = strip_secrets(c_meta["api"])
    meta = {"model": c_meta, "global": g, "judge": strip_secrets(judge),
            "vllm_version": vllm_version(g) if c["backend"] == "vllm" else None,
            "start_time": t_start.isoformat(), "jobs": []}

    modes = c["modes"]
    repeats = int(g["_repeats_override"] or c.get("repeats", g["repeats"]))
    todo = [(m, r) for m in modes for r in range(1, repeats + 1)]
    pending = [(m, r) for (m, r) in todo if not job_paths(g, judge, c, m, r)["accuracy"].exists()]
    need_server = any(not prediction_complete(job_paths(g, judge, c, m, r)) for (m, r) in pending)

    if args.stage == "evaluation":
        need_server = False
    serve = c["backend"] == "vllm" and need_server and not args.no_serve
    gpus = None
    if serve:
        n_gpus = c.get("num_gpus", g.get("default_num_gpus", "all"))
        n_gpus = len(pools["gpu"].all) if str(n_gpus) == "all" else int(n_gpus)
        gpus = pools["gpu"].acquire(n_gpus, fixed=c.get("gpus"), tag=name)
        port = int(c["port"]) if c.get("port") else pools["port"].acquire()
        backend = make_backend(g, c, log_dir, args.dry_run, gpus=",".join(gpus), port=port)
        meta["gpus"], meta["port"] = gpus, port
    else:
        backend = make_backend(g, c, log_dir, args.dry_run, gpus=str(c.get("gpus", "")), port=c.get("port", 0))
    released = threading.Event()
    try:
        if serve:
            for attempt in range(2):
                backend.start()
                try:
                    backend.wait_ready()
                    break
                except RuntimeError as e:
                    if attempt == 0:
                        log(name, f"vLLM failed to start ({str(e)[:120]}), retrying once in 30s")
                        backend.stop()
                        time.sleep(30)
                    else:
                        raise
            backend.smoke_test(args.dry_run)
        elif need_server and c["backend"] == "api":
            log(name, f"backend=api, base_url={backend.base_url}")
            backend.smoke_test(args.dry_run)
        elif not need_server:
            log(name, "all predictions exist or stage=evaluation: no model server needed")
        # One thread per (mode, run): prediction -> evaluation (stage=all) without waiting for the others.
        # vLLM is stopped as soon as every prediction finished (evaluations keep running on the judge API).
        pred_done = threading.Semaphore(0)

        def job(m, r):
            try:
                st = run_prediction_job(g, judge, c, m, modes[m], r, backend, args, log_dir)
            finally:
                pred_done.release()
            return run_evaluation_job(g, judge, c, st, eval_sem, args, log_dir)

        def close_server_when_predictions_done():
            for _ in todo:
                pred_done.acquire()
            if serve:
                log(name, "all predictions finished, stopping vLLM to free the GPUs (evaluation continues)")
                backend.stop()
                pools["gpu"].release(gpus)
                meta["gpu_released_time"] = datetime.now().isoformat()
            released.set()

        closer = threading.Thread(target=close_server_when_predictions_done, daemon=True)
        closer.start()
        # (mode x run) jobs of one model run concurrently by default; `max_parallel_jobs` in the model config caps
        # them (e.g. 1-3 for external APIs with limited throughput)
        n_par = int(c.get("max_parallel_jobs", len(todo)) or len(todo))
        with ThreadPoolExecutor(max_workers=max(1, min(len(todo), n_par))) as ex:
            futs = [ex.submit(job, m, r) for (m, r) in todo]
            meta["jobs"] = [f.result() for f in futs]
        closer.join(timeout=600)
    except Exception as e:
        meta["error"] = f"{e}\n{traceback.format_exc()}"
        log(name, f"model-level error: {e}")
    finally:
        if serve and not released.is_set():      # error path: make sure the server is gone
            backend.stop()
            pools["gpu"].release(gpus)
            meta["gpu_released_time"] = datetime.now().isoformat()
    try:
        meta["end_time"] = datetime.now().isoformat()
        with open(meta_dir / f"run_metadata_{t_start.strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        log(name, f"failed to write metadata: {e}")
    return meta


def busy_gpus(threshold_mib=2000):
    """GPUs whose memory usage exceeds the threshold (owned by somebody else; excluded from the pool)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {l.split(",")[0].strip() for l in out.strip().splitlines()
                if l.strip() and int(l.split(",")[1]) > threshold_mib}
    except Exception:
        return set()


# ----------------------------------------------------------------------------- preflight
def preflight(g, judge, models, args):
    problems = []
    if not Path(g["eval_python"]).exists():
        problems.append(f"eval_python not found: {g['eval_python']}")
    vllm_models = [c for c in models if c["backend"] == "vllm"]
    if vllm_models and not args.no_serve and not args.dry_run and not Path(g["vllm_bin"]).exists():
        problems.append(f"vllm executable not found ({g['vllm_bin']}); install vLLM or set vllm_bin in configs/global.yaml")
    for c in models:
        for k in ("model_name", "modes"):
            if k not in c:
                problems.append(f"{c.get('_config_path')}: missing `{k}`")
        if c["backend"] == "vllm":
            for k in ("model_path", "max_model_len"):
                if k not in c:
                    problems.append(f"{c['model_name']}: backend=vllm requires `{k}`")
            pool = [x.strip() for x in str(g.get("gpu_pool", "")).split(",") if x.strip()]
            if not pool and not args.dry_run and not args.no_serve and args.stage != "evaluation":
                problems.append("no GPU available: set gpu_pool in configs/global.yaml or CUDA_VISIBLE_DEVICES")
            if c.get("gpus") and not all(x.strip() in pool for x in str(c["gpus"]).split(",")):
                problems.append(f"{c['model_name']}: fixed gpus={c['gpus']} not in gpu_pool {pool}")
            if args.no_serve and not c.get("port"):
                problems.append(f"{c['model_name']}: --no-serve requires `port` in the model config")
            for scope, cfg in [("", c)] + [(f"/{m}", mc) for m, mc in c.get("modes", {}).items()]:
                if cfg.get("system_prompt_prefix_json") and not Path(cfg["system_prompt_prefix_json"]).exists():
                    problems.append(f"{c['model_name']}{scope}: system_prompt_prefix_json not found: {cfg['system_prompt_prefix_json']}")
            vargs = [str(a) for a in c.get("vllm_args", [])]
            for a in vargs:
                parts = shlex.split(a)
                if parts[:1] == ["--tokenizer"] and len(parts) > 1 and "$" not in parts[1]:
                    tok = Path(parts[1])
                    if tok.is_dir() and not (tok / "tokenizer.json").exists():
                        problems.append(f"{c['model_name']}: {tok}/tokenizer.json is missing; see the comment in "
                                        f"{Path(c['_config_path']).name} for the symlink to create")
            mp = str(c.get("model_path", ""))
            if "$" in mp:
                problems.append(f"{c['model_name']}: unresolved variable in model_path `{mp}` "
                                f"(export MODELS_DIR=/path/to/models, see .env.example)")
            elif mp and not Path(mp).exists() and not args.dry_run and args.stage != "evaluation":
                problems.append(f"{c['model_name']}: model_path not found: {mp}")
        else:
            if not c["api"].get("base_url"):
                problems.append(f"{c['model_name']}: backend=api requires api.base_url")
        for m, mc in c.get("modes", {}).items():
            for k in ("temperature", "max_tokens"):
                if k not in mc:
                    problems.append(f"{c['model_name']}/{m}: mode is missing `{k}`")
    ports = [c["port"] for c in vllm_models if c.get("port")]
    if len(ports) != len(set(ports)):
        problems.append(f"port conflict: {ports}")
    if args.stage != "prediction" and not args.dry_run:
        if not judge["api_key"]:
            problems.append(f"judge API key is empty: export {judge.get('api_key_env') or 'OPENAI_API_KEY'} "
                            f"or set api_key in {judge['_config_path']}")
        else:
            try:
                r = requests.get(f"{judge['base_url'].rstrip('/')}/models",
                                 headers={"Authorization": f"Bearer {judge['api_key']}"}, timeout=15)
                if r.status_code != 200:
                    problems.append(f"judge authentication failed ({r.status_code}): {r.text[:120]} -> check "
                                    f"{judge.get('api_key_env') or 'api_key'} / {judge['_config_path']}")
            except Exception as e:
                problems.append(f"judge endpoint unreachable {judge['base_url']}: {e}")
    if problems:
        print("\nPreflight failed:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    log("preflight", "ok")


def list_configs():
    print("Model configs (configs/*.yaml):")
    for p in list_yaml(CONFIG_DIR):
        if Path(p).name == "global.yaml":
            continue
        c = load_yaml(p)
        flag = "" if c.get("enabled", True) else "  [enabled=false]"
        print(f"  {Path(p).stem:<36} backend={c.get('backend', 'vllm'):<5} modes={list((c.get('modes') or {}).keys())}{flag}")
    print("Judge configs (judge_configs/*.yaml):")
    for p in list_yaml(JUDGE_DIR):
        j = load_yaml(p)
        print(f"  {Path(p).stem:<36} backbone={j.get('backbone', Path(p).stem)} T={j.get('temperature', 1.0)} "
              f"effort={j.get('reasoning_effort', 'none')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="only run these models (config file stem or model_name); explicitly listed models run even if enabled=false")
    ap.add_argument("--judge", default=None, help="judge config name (judge_configs/<name>.yaml); default: `judge` in global.yaml")
    ap.add_argument("--stage", choices=["all", "prediction", "evaluation"], default="all",
                    help="all = prediction + evaluation; prediction = predictions only; evaluation = score existing predictions")
    ap.add_argument("--repeats", type=int, default=None, help="override the number of independent runs per mode")
    ap.add_argument("--input-dir", default=None, help="override input_dir of global.yaml (e.g. data/smoke for a quick test)")
    ap.add_argument("--results-root", default=None, help="override results_root of global.yaml")
    ap.add_argument("--no-serve", action="store_true", help="do not start vLLM, reuse a server already listening on `port`")
    ap.add_argument("--sequential", action="store_true", help="run models one after another (default: parallel within the GPU pool)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running anything")
    ap.add_argument("--list", action="store_true", help="list model / judge configs and exit")
    args = ap.parse_args()
    if args.list:
        list_configs()
        return

    os.chdir(AGENTIF_DIR)
    g = resolve_environment(load_yaml(CONFIG_DIR / "global.yaml"))
    if args.input_dir:
        g["input_dir"] = args.input_dir
    if args.results_root:
        g["results_root"] = args.results_root
    g["_repeats_override"] = args.repeats
    if args.repeats:
        g["repeats"] = args.repeats
    judge = load_judge(args.judge or g["judge"])
    models = load_models(args.models)
    if not models:
        sys.exit("no model config to run (see --list)")
    log_dir = AGENTIF_DIR / g["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    (AGENTIF_DIR / g["cache_dir"]).mkdir(parents=True, exist_ok=True)
    preflight(g, judge, models, args)
    log("main", f"models: {[(c['model_name'], c['backend']) for c in models]}, repeats={g['repeats']}, "
                f"judge={judge['name']} (backbone={judge['backbone']}, T={judge['temperature']}, "
                f"max_tokens={judge['max_tokens']}, effort={judge['reasoning_effort']})")

    eval_sem = threading.BoundedSemaphore(int(judge["max_concurrent_jobs"]))
    pool_gpus = [x.strip() for x in str(g.get("gpu_pool", "")).split(",") if x.strip()]
    busy = busy_gpus()
    if busy & set(pool_gpus):
        log("main", f"GPU {sorted(busy & set(pool_gpus), key=int)} already in use (> 2 GB), removed from the pool")
        pool_gpus = [x for x in pool_gpus if x not in busy]
    pools = {"gpu": GPUPool(pool_gpus), "port": PortPool(g.get("port_base", 8101))}
    log("main", f"stage={args.stage}, gpu_pool={pools['gpu'].free}")
    if args.sequential:
        metas = [run_model(g, judge, c, args, eval_sem, log_dir, pools) for c in models]
    else:
        # one thread per model; vllm models queue in GPUPool.acquire and run in parallel when GPUs allow
        with ThreadPoolExecutor(max_workers=len(models)) as ex:
            metas = list(ex.map(lambda c: run_model(g, judge, c, args, eval_sem, log_dir, pools), models))

    print("\n================ job status ================")
    for m in metas:
        for j in m["jobs"]:
            print(f"{m['model']['model_name']:<32} {j['mode']:<13} run{j['run']}  pred={j['prediction']}  eval={j['evaluation']}"
                  + (f"  CSR={j['CSR']:.4f} ISR={j['ISR']:.4f}" if j.get("CSR") is not None else ""))
        if m.get("error"):
            print(f"{m['model']['model_name']}: ERROR {m['error'].splitlines()[0]}")
    if JUDGE_DOWN.is_set():
        print("\n!!! the judge endpoint was unreachable; fix it and re-run with --stage evaluation "
              "(finished jobs are skipped automatically)")
    if not args.dry_run and args.stage != "prediction":
        subprocess.run([sys.executable, str(CODE_DIR / "aggregate.py"), "--results_root", g["results_root"]],
                       cwd=str(AGENTIF_DIR))


if __name__ == "__main__":
    main()
