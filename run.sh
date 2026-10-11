#!/usr/bin/env bash
# AgentIF one-command evaluation.
#
#   1. cp .env.example .env   and fill in OPENAI_API_KEY (judge) and MODELS_DIR (local model weights)
#   2. add / edit a model config in configs/<model>.yaml
#   3. bash run.sh                                   # every enabled model: prediction + evaluation
#      bash run.sh --models Qwen3.5-4B gemma-4-E4B-it
#      bash run.sh --stage prediction                # predictions only
#      bash run.sh --stage evaluation                # score existing predictions (no GPU needed)
#      bash run.sh --repeats 1 --input-dir data/smoke --results-root results_smoke   # 5-instruction smoke test
#      bash run.sh --list                            # show available model / judge configs
#
# Run in the background for the full benchmark:
#   nohup bash run.sh > logs/run.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
mkdir -p logs
exec python code4eval/batch_eval.py "$@"
