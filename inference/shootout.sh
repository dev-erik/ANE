#!/usr/bin/env bash
set -euo pipefail

# Inference Engine Shootout for Qwen2.5-0.5B on M4 Max
# Benchmarks: Raw MLX, llama.cpp, Our engine (CPU AMX)
# All Q4 quantized where possible, 200 token generation

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/shootout_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/shootout_${TIMESTAMP}.json"
LOG_FILE="${RESULTS_DIR}/shootout_${TIMESTAMP}.log"

PROMPT_SHORT="Explain quantum computing in simple terms."
MAX_TOKENS=200
N_RUNS=5

GGUF_MODEL="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MLX_MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LLAMA_BENCH="/tmp/llama_cpp_build/build/bin/llama-bench"
LLAMA_CLI="/tmp/llama_cpp_build/build/bin/llama-cli"
OUR_ENGINE="${SCRIPT_DIR}/qwen_ane"
OUR_WEIGHTS="${SCRIPT_DIR}/qwen05b.bin"

echo "========================================" | tee "$LOG_FILE"
echo " Inference Engine Shootout" | tee -a "$LOG_FILE"
echo " Qwen2.5-0.5B on M4 Max (128GB)" | tee -a "$LOG_FILE"
echo " $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "--- 1/3: Raw MLX (Q4) ---" | tee -a "$LOG_FILE"
python3 -c "
import time, json, sys
import mlx_lm

model, tokenizer = mlx_lm.load('${MLX_MODEL}')

results = []
for run in range(${N_RUNS}):
    resp = mlx_lm.generate(model, tokenizer, prompt='${PROMPT_SHORT}', max_tokens=${MAX_TOKENS}, verbose=True)
    sys.stdout.flush()

print('MLX_DONE')
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "--- 2/3: llama.cpp (Q4_K_M) ---" | tee -a "$LOG_FILE"
if [ -f "$LLAMA_BENCH" ]; then
    "$LLAMA_BENCH" -m "$GGUF_MODEL" -p 8 -n "$MAX_TOKENS" -r "$N_RUNS" -ngl 99 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    "$LLAMA_BENCH" -m "$GGUF_MODEL" -p 298 -n "$MAX_TOKENS" -r "$N_RUNS" -ngl 99 2>&1 | tee -a "$LOG_FILE"
else
    echo "SKIP: llama-bench not found at $LLAMA_BENCH" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "--- 3/3: Our Engine (CPU AMX, F16) ---" | tee -a "$LOG_FILE"
if [ -f "$OUR_ENGINE" ]; then
    for run in $(seq 1 "$N_RUNS"); do
        echo "$PROMPT_SHORT" | "$OUR_ENGINE" "$OUR_WEIGHTS" --max-tokens "$MAX_TOKENS" 2>&1 | grep -E "Prefill|Decode" | tee -a "$LOG_FILE"
    done
else
    echo "SKIP: our engine not found at $OUR_ENGINE" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo " Shootout complete. Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
