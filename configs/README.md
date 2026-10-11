# Model configs

One YAML file per model. `bash run.sh` runs every file in this directory (except `*.example.yaml` and files with
`enabled: false`); `bash run.sh --models <file stem> ...` runs a selection. `global.yaml` holds the settings shared
by all models and is described at the end.

## Minimal templates

Local model served by vLLM:

```yaml
backend: vllm
model_name: my-model                       # result directory name
model_path: ${MODELS_DIR}/my-model         # local path (MODELS_DIR comes from .env) or a HF repo id
max_model_len: 65536                       # context window to serve; must cover prompt (<= ~20K tokens) + max_tokens
vllm_args:                                 # passed to `vllm serve` verbatim
  - --dtype bfloat16
  - --gpu-memory-utilization 0.85
  - --reasoning-parser qwen3               # whatever parser strips this model's thinking block
modes:
  thinking:
    enable_thinking: true
    temperature: 1.0
    top_p: 0.95
    max_tokens: 32768
  non_thinking:
    enable_thinking: false
    temperature: 0.0
    top_p: 0.95
    max_tokens: 16384
```

Model behind an OpenAI-compatible API (no deployment):

```yaml
backend: api
model_name: my-api-model
api:
  base_url: https://api.example.com/v1
  model: my-api-model                      # `model` field of the request
  api_key_env: MODEL_API_KEY               # key is read from this environment variable (put it in .env)
modes:
  non_thinking:
    temperature: 0.0
    max_tokens: 16384
  thinking:
    temperature: 1.0
    max_tokens: 32768
    extra_body: {"thinking": {"type": "enabled"}}   # vendor-specific switch, if any
```

Try a new config with the 5-instruction smoke set before the full run:

```bash
bash run.sh --models my-model --repeats 1 --input-dir data/smoke --results-root results_smoke
```

## Field reference

| Key | Required | Default | Meaning |
|---|---|---|---|
| `model_name` | no | file stem | Name used in `results/prediction/<model_name>__<mode>/` and in the summary |
| `backend` | no | `vllm` | `vllm` = start a local server; `api` = call an existing OpenAI-compatible endpoint |
| `enabled` | no | `true` | `false` = skipped unless listed explicitly with `--models` |
| `repeats` | no | `global.yaml` | Independent sampling runs per mode for this model |
| `prediction_num_workers` | no | `global.yaml` | Concurrent requests per prediction job; lower it (e.g. 4) for rate-limited external APIs |
| `max_parallel_jobs` | no | all | Cap on simultaneous (mode × run) jobs of this model; use 1-3 for external APIs with limited throughput |
| `modes` | **yes** | | One entry per mode to evaluate (see below). Single-mode models list one entry |
| **backend = vllm** | | | |
| `model_path` | **yes** | | Path or HF repo id given to `vllm serve`. `${VAR}` is expanded from the environment |
| `max_model_len` | **yes** | | `--max-model-len`; also used to trim `max_tokens` for long prompts |
| `served_model_name` | no | `model_name` | `--served-model-name` and the `model` field of requests |
| `tensor_parallel_size` | no | `1` | `--tensor-parallel-size` |
| `data_parallel_size` | no | GPUs ÷ TP | `--data-parallel-size`; by default every allocated GPU hosts one replica |
| `num_gpus` | no | `default_num_gpus` | GPUs to take from the pool; smaller values let several models run in parallel |
| `gpus` | no | | Fixed GPU ids (e.g. `"0,1"`) instead of taking any free ones from the pool |
| `port` | no | auto | Fixed port; required with `--no-serve` (reuse a server you started yourself) |
| `vllm_args` | no | `[]` | Extra `vllm serve` flags, one string per list item, passed verbatim |
| `request_mode` | no | | `raw_harmony` for gpt-oss models: the prediction script calls `/tokenize` + `/v1/completions` and parses the Harmony channels itself (vLLM's chat endpoint drops answers sent on the `commentary` channel) |
| `system_prompt_prefix_json` | no | | JSON `{"role": "system", "content": ...}` prepended to every instruction when the vendor requires a fixed system prompt (see `assets/`). Can also be set inside a mode, e.g. `/no_think` for Nemotron |
| **backend = api** | | | |
| `api.base_url` | **yes** | | Endpoint; `api.base_url_env` names an environment variable that overrides it |
| `api.model` | no | `model_name` | `model` field of the request |
| `api.api_key` / `api.api_key_env` | no | `EMPTY` | Literal key, or the environment variable holding it (preferred) |
| `api.max_model_len` | no | | Enables prompt-length based trimming of `max_tokens` (needs a vLLM `/tokenize` endpoint) |

Keys inside each `modes.<name>` entry (the mode name is free, `thinking` / `non_thinking` are the convention):

| Key | Required | Meaning |
|---|---|---|
| `temperature`, `max_tokens` | **yes** | Sampling temperature and output budget |
| `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` | no | Not sent when omitted (server default). `top_k` / `min_p` / `repetition_penalty` go through vLLM's `extra_body` |
| `enable_thinking` | no | Sends `chat_template_kwargs: {enable_thinking: ...}`. Only for models whose chat template supports it; omit for Mistral-format models and most external APIs |
| `extra_body` | no | Any JSON merged into the request body, e.g. a vendor thinking switch, `reasoning_effort`, or `{"skip_special_tokens": false}` |
| `system_prompt_prefix_json` | no | Mode-level override of the model-level key (see above) |

`assets/` holds the small files some configs refer to: the Ministral reasoning system prompt, the `/no_think` prefix for Nemotron-Nano-9B-v2, and a corrected `tokenizer_config.json` for DeepSeek-R1-0528-Qwen3-8B (that one needs a symlink to the model's `tokenizer.json`, see the comment in its config).

## Conventions used for the leaderboard

- **Both modes** of a hybrid model are evaluated and reported as separate rows. Dedicated thinking-only or
  non-thinking-only checkpoints get a single mode.
- **Sampling parameters follow the vendor's recommendation** for the model (and mode) when one exists. Otherwise the
  benchmark defaults apply: thinking `T=1.0, top_p=0.95, max_tokens=32768`; non-thinking `T=0.0, top_p=0.95,
  max_tokens=16384`.
- **Do not add `--enable-auto-tool-choice` / `--tool-call-parser`.** AgentIF prompts specify a textual tool-call
  format; a tool-call parser would swallow native tool-call tokens and return an empty response. A
  `--reasoning-parser` is fine and should be used to strip thinking blocks.
- Write down the source of the parameters in a comment at the top of the file, as the shipped configs do.

## global.yaml

Shared settings; most users only ever touch `repeats` and perhaps `gpu_pool`.

| Key | Default | Meaning |
|---|---|---|
| `eval_python` | `""` | Python used for `code4eval/*.py`. Empty = the interpreter running `batch_eval.py` |
| `vllm_bin` | `""` | `vllm` executable. Empty = the one found on `PATH` |
| `judge` | `gpt-5.6-luna` | Default judge, `judge_configs/<judge>.yaml`; override per run with `--judge` |
| `gpu_pool` | `auto` | GPUs vLLM may use. `auto` = all GPUs from `nvidia-smi` (or `CUDA_VISIBLE_DEVICES` if set); or an explicit list `"0,1,2,3"`. GPUs with more than 2 GB already in use are dropped at start-up |
| `default_num_gpus` | `all` | GPUs a model takes unless its config sets `num_gpus`. `all` = the whole pool, so models run one after another with one replica per GPU |
| `port_base` | `8101` | First port tried for vLLM servers |
| `repeats` | `3` | Independent runs per (model, mode); `--repeats` overrides |
| `input_dir` | `data/` | Directory containing `eval.json`; `--input-dir` overrides (e.g. `data/smoke`) |
| `results_root` | `results` | Output root; `--results-root` overrides |
| `log_dir` | `logs` | vLLM, prediction and evaluation logs |
| `cache_dir` | `.cache` | Response caches (`.pkl`) of prediction and judge calls |
| `prediction.num_workers` | `32` | Concurrent requests per prediction job; all (mode × run) jobs of a model share one server |
| `serve.health_timeout_sec` | `2400` | Maximum wait for a vLLM server to become healthy |
| `serve.shutdown_grace_sec` | `60` | Time given to vLLM to exit on SIGTERM before SIGKILL |

Judge-side parameters (endpoint, temperature, `reasoning_effort`, requests in flight, concurrent evaluation jobs)
live in `judge_configs/<name>.yaml`; keep them unchanged for numbers comparable to the leaderboard.
