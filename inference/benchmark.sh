#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present (LMS_API_KEY, LMS_PORT, LMS_MODEL)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

BINARY="$SCRIPT_DIR/qwen_ane"
WEIGHTS="$SCRIPT_DIR/qwen05b.bin"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen2.5-0.5B-Instruct}"
SOCK="/tmp/qwen_ane_bench.sock"
HTTP_PORT=8877
RESULTS_JSON="$SCRIPT_DIR/benchmark_results.json"

# --- Prompt suite ---
PROMPT_NAMES=(   "tiny"   "short"          "medium"                                                 "long"                                                                      "stress")
PROMPTS=(        "Hi"     "What is 2+2?"   "Explain how neural networks work in 3 sentences."       "Write a short story about a robot learning to paint. Include dialogue."     "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.")
MAX_TOKENS=(     10       20               100                                                      200                                                                         50)

info()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
dim()   { printf "\033[2m%s\033[0m\n" "$1"; }

# Extract a numeric or string value from flat JSON. No python needed.
# Usage: json_val '{"key":123}' "key"  →  123
json_val() {
    local json="$1" key="$2"
    echo "$json" | sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^,\"}\]*\)\"\{0,1\}.*/\1/p" | head -1
}

# Extract the "text" field which may contain escaped chars and commas.
# Grabs everything between "text":" and the next unescaped quote.
json_text() {
    local json="$1"
    echo "$json" | sed -n 's/.*"text":"\(.*\)","prompt_tokens".*/\1/p' | sed 's/\\n/ /g; s/\\"//g'
}

# Truncate a float string to integer: "317.2" → "317"
trunc() { echo "${1%%.*}"; }

# Average an array of numbers using awk. Handles both ints and floats.
# Usage: shell_avg "1.5" "2.3" "3.1"  →  2.3
shell_avg() { printf '%s\n' "$@" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "0"}'; }
shell_avg_int() { printf '%s\n' "$@" | awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "0"}'; }

# --- Preflight ---
if [ ! -f "$BINARY" ]; then
    echo "Binary not found: $BINARY"
    echo "Run setup.sh first: $SCRIPT_DIR/setup.sh"
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "Weights not found: $WEIGHTS"
    echo "Run setup.sh first: $SCRIPT_DIR/setup.sh"
    exit 1
fi

# Detect hardware
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
MACOS=$(sw_vers -productVersion 2>/dev/null || echo "Unknown")
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
MEM_GB=$((MEM_BYTES / 1073741824))

echo ""
info "=== ANE Multi-Format Inference Benchmark ==="
echo "Hardware: $CHIP"
echo "macOS:    $MACOS"
echo "Memory:   ${MEM_GB} GB"
echo "Model:    Qwen2.5-0.5B-Instruct (494M params)"
echo ""

# --- Phase 0: Prepare weight files (F16 + Q8) ---
WEIGHTS_F16="$SCRIPT_DIR/qwen05b.bin"
WEIGHTS_Q8="$SCRIPT_DIR/qwen05b_q8.bin"
WEIGHTS_Q4="$SCRIPT_DIR/qwen05b_q4.bin"
CONVERT="$SCRIPT_DIR/convert_weights.py"
VENV_DIR="$SCRIPT_DIR/.venv"

info "Phase 0: Preparing weight files"

if [ ! -f "$WEIGHTS_Q8" ]; then
    if [ ! -f "$CONVERT" ]; then
        echo "  convert_weights.py not found, skipping Q8 generation."
        WEIGHTS_Q8=""
    else
        dim "Generating Q8 weights (one-time)..."
        if [ -d "$VENV_DIR" ]; then
            source "$VENV_DIR/bin/activate"
        fi
        python3 "$CONVERT" "$MODEL_DIR" "$WEIGHTS_Q8" --q8
        dim "Q8 weights ready: $(du -h "$WEIGHTS_Q8" | cut -f1)"
    fi
else
    dim "Q8 weights already exist: $(du -h "$WEIGHTS_Q8" | cut -f1)"
fi

if [ ! -f "$WEIGHTS_Q4" ]; then
    if [ ! -f "$CONVERT" ]; then
        echo "  convert_weights.py not found, skipping Q4 generation."
        WEIGHTS_Q4=""
    else
        dim "Generating Q4 weights (one-time)..."
        if [ -d "$VENV_DIR" ]; then
            source "$VENV_DIR/bin/activate"
        fi
        python3 "$CONVERT" "$MODEL_DIR" "$WEIGHTS_Q4" --q4
        dim "Q4 weights ready: $(du -h "$WEIGHTS_Q4" | cut -f1)"
    fi
else
    dim "Q4 weights already exist: $(du -h "$WEIGHTS_Q4" | cut -f1)"
fi

dim "F16 weights: $(du -h "$WEIGHTS_F16" | cut -f1)"
echo ""

# ANE weight formats to benchmark
# GPU flag: empty for CPU formats, "--gpu" for Metal GPU formats
ANE_FMT_NAMES=("F16")
ANE_FMT_WEIGHTS=("$WEIGHTS_F16")
ANE_FMT_LABELS=("F16→F32 (AMX)")
ANE_FMT_GPU=("")

if [ -n "$WEIGHTS_Q8" ] && [ -f "$WEIGHTS_Q8" ]; then
    ANE_FMT_NAMES+=("Q8")
    ANE_FMT_WEIGHTS+=("$WEIGHTS_Q8")
    ANE_FMT_LABELS+=("Q8 (NEON dequant)")
    ANE_FMT_GPU+=("")
fi

if [ -n "$WEIGHTS_Q4" ] && [ -f "$WEIGHTS_Q4" ]; then
    ANE_FMT_NAMES+=("Q4_Metal")
    ANE_FMT_WEIGHTS+=("$WEIGHTS_Q4")
    ANE_FMT_LABELS+=("Q4 SIMD (Metal GPU)")
    ANE_FMT_GPU+=("--gpu")

    ANE_FMT_NAMES+=("Q4_AMX")
    ANE_FMT_WEIGHTS+=("$WEIGHTS_Q4")
    ANE_FMT_LABELS+=("Q4→F32 (AMX dequant)")
    ANE_FMT_GPU+=("")
fi

NUM_ANE_FMTS=${#ANE_FMT_NAMES[@]}
NUM_PROMPTS=${#PROMPTS[@]}

# Global cleanup
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    rm -f "$SOCK" /tmp/qwen_bench_server.log
}
trap cleanup EXIT

# Helper: start server with given weight file and optional extra flags, wait for READY
start_server() {
    local wfile="$1"
    shift
    local extra_flags="$*"
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    sleep 1
    rm -f /tmp/qwen_bench_server.log
    "$BINARY" "$wfile" --http "$HTTP_PORT" --model-dir "$MODEL_DIR" $extra_flags > /tmp/qwen_bench_server.log 2>&1 &
    SERVER_PID=$!
    for _i in $(seq 1 30); do
        if grep -q "READY" /tmp/qwen_bench_server.log 2>/dev/null; then return 0; fi
        sleep 1
    done
    echo "Server failed to start with $wfile. Log:"
    cat /tmp/qwen_bench_server.log
    return 1
}

# --- Phase 1: Multi-format ANE benchmarks ---
# Per-format result tracking (indexed by format number)
declare -a ALL_AVG_P ALL_AVG_D ALL_AVG_INF ALL_AVG_TTFT ALL_AVG_RT
ANE_JSON_BLOCKS=""

for fmt_idx in $(seq 0 $((NUM_ANE_FMTS - 1))); do
    FMT_NAME="${ANE_FMT_NAMES[$fmt_idx]}"
    FMT_WEIGHTS="${ANE_FMT_WEIGHTS[$fmt_idx]}"
    FMT_LABEL="${ANE_FMT_LABELS[$fmt_idx]}"
    FMT_GPU="${ANE_FMT_GPU[$fmt_idx]}"

    echo ""
    info "Phase 1.$((fmt_idx+1)): ANE $FMT_NAME benchmark ($FMT_LABEL)"
    dim "Weights: $(du -h "$FMT_WEIGHTS" | cut -f1) — Starting server..."

    if ! start_server "$FMT_WEIGHTS" $FMT_GPU; then
        echo "Skipping $FMT_NAME format."
        ALL_AVG_P+=("0"); ALL_AVG_D+=("0"); ALL_AVG_INF+=("0")
        ALL_AVG_TTFT+=("0"); ALL_AVG_RT+=("0")
        continue
    fi

    dim "Warmup run (discarded)..."
    curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"warmup","max_tokens":5}' > /dev/null 2>&1
    echo ""

    printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" \
        "Prompt" "In" "Out" "Prefill" "Decode" "TTFT" "Infer" "Rndtrip" "Overhead"
    printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" \
        "" "tok" "tok" "(t/s)" "(t/s)" "(ms)" "(ms)" "(ms)" "(ms)"
    printf '%.0s─' {1..85}; echo ""

    declare -a P_TPS_ARR=() D_TPS_ARR=() INF_MS_ARR=() TTFT_MS_ARR=() RT_MS_ARR=()
    FMT_JSON_ENTRIES=""

    for i in $(seq 0 $((NUM_PROMPTS - 1))); do
        NAME="${PROMPT_NAMES[$i]}"
        PROMPT="${PROMPTS[$i]}"
        MAXTOK="${MAX_TOKENS[$i]}"

        RT_T0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
        RESP=$(curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAXTOK}" 2>&1)
        RT_T1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
        RT_MS=$(echo "$RT_T0 $RT_T1" | awk '{printf "%.0f", ($2 - $1) * 1000}')

        P_TOKENS=$(json_val "$RESP" "prompt_tokens")
        G_TOKENS=$(json_val "$RESP" "gen_tokens")
        P_TPS=$(json_val "$RESP" "prefill_tps")
        D_TPS=$(json_val "$RESP" "decode_tps")
        TTFT_MS=$(trunc "$(json_val "$RESP" "ttft_ms")")
        INF_MS=$(trunc "$(json_val "$RESP" "inference_ms")")
        TOTAL_MS=$(trunc "$(json_val "$RESP" "total_ms")")
        TEXT=$(json_text "$RESP")
        OVERHEAD=$((RT_MS - TOTAL_MS))

        printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" \
            "$NAME" "$P_TOKENS" "$G_TOKENS" "$P_TPS" "$D_TPS" "$TTFT_MS" "$INF_MS" "$RT_MS" "$OVERHEAD"

        P_TPS_ARR+=("$P_TPS")
        D_TPS_ARR+=("$D_TPS")
        INF_MS_ARR+=("$INF_MS")
        TTFT_MS_ARR+=("$TTFT_MS")
        RT_MS_ARR+=("$RT_MS")

        FMT_JSON_ENTRIES="$FMT_JSON_ENTRIES{\"name\":\"$NAME\",\"prompt_tokens\":$P_TOKENS,\"gen_tokens\":$G_TOKENS,\"prefill_tps\":$P_TPS,\"decode_tps\":$D_TPS,\"ttft_ms\":$TTFT_MS,\"inference_ms\":$INF_MS,\"roundtrip_ms\":$RT_MS},"

        echo "    → $TEXT"
        echo ""
    done

    printf '%.0s─' {1..85}; echo ""

    F_AVG_P=$(shell_avg "${P_TPS_ARR[@]}")
    F_AVG_D=$(shell_avg "${D_TPS_ARR[@]}")
    F_AVG_INF=$(shell_avg_int "${INF_MS_ARR[@]}")
    F_AVG_TTFT=$(shell_avg_int "${TTFT_MS_ARR[@]}")
    F_AVG_RT=$(shell_avg_int "${RT_MS_ARR[@]}")
    F_AVG_OVERHEAD=$((F_AVG_RT - F_AVG_INF))
    printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" "Average" "" "" "$F_AVG_P" "$F_AVG_D" "$F_AVG_TTFT" "$F_AVG_INF" "$F_AVG_RT" "$F_AVG_OVERHEAD"
    echo ""

    ALL_AVG_P+=("$F_AVG_P")
    ALL_AVG_D+=("$F_AVG_D")
    ALL_AVG_INF+=("$F_AVG_INF")
    ALL_AVG_TTFT+=("$F_AVG_TTFT")
    ALL_AVG_RT+=("$F_AVG_RT")

    ANE_JSON_BLOCKS="$ANE_JSON_BLOCKS
    \"$FMT_NAME\": {
      \"format\": \"$FMT_NAME\",
      \"label\": \"$FMT_LABEL\",
      \"weight_size_mb\": $(du -m "$FMT_WEIGHTS" | cut -f1),
      \"avg_prefill_tps\": $F_AVG_P,
      \"avg_decode_tps\": $F_AVG_D,
      \"avg_inference_ms\": $F_AVG_INF,
      \"avg_roundtrip_ms\": $F_AVG_RT,
      \"avg_ttft_ms\": $F_AVG_TTFT,
      \"results\": [${FMT_JSON_ENTRIES%,}]
    },"
done

# Use F16 results as the primary ANE numbers (first format)
AVG_P="${ALL_AVG_P[0]}"
AVG_D="${ALL_AVG_D[0]}"
AVG_INF="${ALL_AVG_INF[0]}"
AVG_TTFT="${ALL_AVG_TTFT[0]}"
AVG_RT="${ALL_AVG_RT[0]}"

info "Infer = server-reported (pure processing). Rndtrip = wall-clock (what clients see)."
echo ""

# --- Phase 2: Cold start measurement ---
info "Phase 2: Cold start (single-shot, recompiles ANE kernels)"

kill "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
sleep 1

COLD_T0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
COLD_OUT=$("$BINARY" "$WEIGHTS" "151644 8948 198 2610 525 264 10950 17847 13 151645 198 151644 872 198 13048 151645 198 151644 77091 198" 10 2>&1 || true)
COLD_T1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
COLD_MS=$(echo "$COLD_T0 $COLD_T1" | awk '{printf "%.0f", ($2 - $1) * 1000}')

echo "Cold start latency: ${COLD_MS}ms (includes ANE kernel compilation)"
echo ""

# Re-start server (F16) for consistency check
start_server "$WEIGHTS_F16"

# --- Phase 3: Repeated prompt (consistency check) ---
info "Phase 3: Decode speed consistency (5x same prompt, F16)"

printf "%-6s %10s %10s %10s\n" "Run" "Prefill" "Decode" "Infer(ms)"
printf '%.0s─' {1..40}; echo ""

for run in $(seq 1 5); do
    RESP=$(curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Count from 1 to 10", "max_tokens": 50}' 2>&1)
    P=$(json_val "$RESP" "prefill_tps")
    D=$(json_val "$RESP" "decode_tps")
    IM=$(trunc "$(json_val "$RESP" "inference_ms")")
    printf "%-6s %10s %10s %10s\n" "#$run" "$P" "$D" "$IM"
done
echo ""

# --- Save JSON results ---
JSON="{
  \"hardware\": \"$CHIP\",
  \"macos\": \"$MACOS\",
  \"memory_gb\": $MEM_GB,
  \"model\": \"Qwen2.5-0.5B-Instruct\",
  \"mode\": \"http_server\",
  \"cold_start_ms\": $COLD_MS,
  \"ane_formats\": {$( echo "$ANE_JSON_BLOCKS" | sed '$ s/,$//' )
  }
}"
echo "$JSON" > "$RESULTS_JSON"
dim "Results saved to $RESULTS_JSON"
echo ""

# --- Phase 4: LM Studio comparison (if running) ---
LMS_PORT="${LMS_PORT:-1234}"
LMS_API_KEY="${LMS_API_KEY:-}"

# Models to benchmark (override via LMS_MODELS env var, comma-separated)
LMS_MODELS_DEFAULT="qwen2.5-0.5b-instruct,qwen2.5-0.5b-instruct-mlx@8bit,qwen2.5-0.5b-instruct-mlx@4bit"
IFS=',' read -ra LMS_MODEL_LIST <<< "${LMS_MODELS:-$LMS_MODELS_DEFAULT}"

# Check if LM Studio is running
LMS_REACHABLE=0
if curl -s --max-time 2 "http://localhost:$LMS_PORT/api/v1/chat" -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1; then
    LMS_REACHABLE=1
fi

if [ "$LMS_REACHABLE" -eq 1 ]; then
    info "Phase 4: LM Studio comparison (localhost:$LMS_PORT)"
    dim "Models: ${LMS_MODEL_LIST[*]}"

    if [ -z "$LMS_API_KEY" ]; then
        echo ""
        echo "  LM Studio requires an API key."
        echo "  Find it in LM Studio > Developer tab > API key"
        echo "  Or set LMS_API_KEY env var before running."
        echo ""
        printf "  Enter LM Studio API key (or press Enter to skip): "
        read -r LMS_API_KEY
        if [ -z "$LMS_API_KEY" ]; then
            dim "Skipping LM Studio benchmark."
            LMS_REACHABLE=0
        fi
    fi
fi

LMS_ALL_JSON=""

if [ "$LMS_REACHABLE" -eq 1 ] && [ -n "$LMS_API_KEY" ]; then

    # Track the best model for the final comparison table
    BEST_LMS_MODEL=""
    BEST_LMS_TPS="0"
    BEST_LMS_LAT="99999"
    BEST_LMS_TTFT="0"

    for LMS_MODEL in "${LMS_MODEL_LIST[@]}"; do
        echo ""
        info "── $LMS_MODEL ──"

        # Test if this model is available
        TEST_RESP=$(curl -s --max-time 10 "http://localhost:$LMS_PORT/api/v1/chat" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $LMS_API_KEY" \
            -d "{\"model\":\"$LMS_MODEL\",\"system_prompt\":\"test\",\"input\":\"hi\"}" 2>&1)

        if echo "$TEST_RESP" | grep -qi "error\|not found\|not loaded\|no model"; then
            dim "  Model '$LMS_MODEL' not available, skipping."
            continue
        fi

        printf "%-10s %5s %5s %10s %10s %10s\n" \
            "Prompt" "In" "Out" "Decode" "TTFT" "Rndtrip"
        printf "%-10s %5s %5s %10s %10s %10s\n" \
            "" "tok" "tok" "(t/s)" "(ms)" "(ms)"
        printf '%.0s─' {1..55}; echo ""

        declare -a LMS_LATENCIES=() LMS_TPS_ARR=() LMS_TTFT_ARR=()
        LMS_JSON_ENTRIES=""

        for i in $(seq 0 $((NUM_PROMPTS - 1))); do
            NAME="${PROMPT_NAMES[$i]}"
            PROMPT="${PROMPTS[$i]}"

            T0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
            LMS_RESP=$(curl -s --max-time 120 "http://localhost:$LMS_PORT/api/v1/chat" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $LMS_API_KEY" \
                -d "{\"model\":\"$LMS_MODEL\",\"system_prompt\":\"You are a helpful assistant. Be concise.\",\"input\":\"$PROMPT\"}" 2>&1)
            T1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
            LMS_MS=$(echo "$T0 $T1" | awk '{printf "%.0f", ($2 - $1) * 1000}')

            eval "$(echo "$LMS_RESP" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    text = r.get('output', [{}])[0].get('content', '').replace(chr(10),' ').replace('\"', '')
    s = r.get('stats', {})
    tps = s.get('tokens_per_second', 0)
    ttft = int(s.get('time_to_first_token_seconds', 0) * 1000)
    in_tok = s.get('input_tokens', 0)
    out_tok = s.get('total_output_tokens', 0)
    print(f'LMS_TEXT=\"{text}\"')
    print(f'LMS_TPS={tps:.1f}')
    print(f'LMS_TTFT={ttft}')
    print(f'LMS_IN={in_tok}')
    print(f'LMS_OUT={out_tok}')
except Exception as e:
    print(f'LMS_TEXT=\"(parse error)\"')
    print('LMS_TPS=0')
    print('LMS_TTFT=0')
    print('LMS_IN=0')
    print('LMS_OUT=0')
" 2>/dev/null)"

            printf "%-10s %5s %5s %10s %10s %10s\n" "$NAME" "$LMS_IN" "$LMS_OUT" "$LMS_TPS" "$LMS_TTFT" "$LMS_MS"
            LMS_LATENCIES+=("$LMS_MS")
            LMS_TPS_ARR+=("$LMS_TPS")
            LMS_TTFT_ARR+=("$LMS_TTFT")
            LMS_JSON_ENTRIES="$LMS_JSON_ENTRIES{\"name\":\"$NAME\",\"latency_ms\":$LMS_MS,\"tps\":$LMS_TPS,\"ttft_ms\":$LMS_TTFT,\"input_tokens\":$LMS_IN,\"output_tokens\":$LMS_OUT},"
        done

        printf '%.0s─' {1..55}; echo ""

        M_AVG_LAT=$(shell_avg_int "${LMS_LATENCIES[@]}")
        M_AVG_TPS=$(shell_avg "${LMS_TPS_ARR[@]}")
        M_AVG_TTFT=$(shell_avg_int "${LMS_TTFT_ARR[@]}")
        printf "%-10s %5s %5s %10s %10s %10s\n" "Average" "" "" "$M_AVG_TPS" "$M_AVG_TTFT" "$M_AVG_LAT"

        # Track the best model by decode t/s
        if awk "BEGIN {exit !($M_AVG_TPS > $BEST_LMS_TPS)}" 2>/dev/null; then
            BEST_LMS_MODEL="$LMS_MODEL"
            BEST_LMS_TPS="$M_AVG_TPS"
            BEST_LMS_LAT="$M_AVG_LAT"
            BEST_LMS_TTFT="$M_AVG_TTFT"
        fi

        LMS_ALL_JSON="$LMS_ALL_JSON
    \"$(echo "$LMS_MODEL" | sed 's/[^a-zA-Z0-9._-]/_/g')\": {
      \"model\": \"$LMS_MODEL\",
      \"avg_latency_ms\": $M_AVG_LAT,
      \"avg_tps\": $M_AVG_TPS,
      \"avg_ttft_ms\": $M_AVG_TTFT,
      \"results\": [${LMS_JSON_ENTRIES%,}]
    },"
    done

    echo ""

    # --- Final Comparison Table: all ANE formats + all LM Studio models ---
    info "=== Multi-Format Comparison ==="
    dim "(All times are wall-clock round-trip, apples-to-apples)"
    echo ""

    # Collect all column names and data
    declare -a COL_NAMES=() COL_DECODE=() COL_PREFILL=() COL_TTFT=() COL_RT=() COL_PREC=() COL_ACCEL=()

    for fi2 in $(seq 0 $((NUM_ANE_FMTS - 1))); do
        COL_NAMES+=("ANE ${ANE_FMT_NAMES[$fi2]}")
        COL_DECODE+=("${ALL_AVG_D[$fi2]}")
        COL_PREFILL+=("${ALL_AVG_P[$fi2]}")
        COL_TTFT+=("${ALL_AVG_TTFT[$fi2]}")
        COL_RT+=("${ALL_AVG_RT[$fi2]}")
        COL_PREC+=("${ANE_FMT_LABELS[$fi2]}")
        if [ -n "${ANE_FMT_GPU[$fi2]}" ]; then
            COL_ACCEL+=("Metal GPU")
        else
            COL_ACCEL+=("CPU (AMX)")
        fi
    done

    # Add each tested LM Studio model as a column
    declare -a LMS_TESTED_NAMES=() LMS_TESTED_TPS=() LMS_TESTED_TTFT=() LMS_TESTED_LAT=()
    for LMS_MODEL in "${LMS_MODEL_LIST[@]}"; do
        # Check if this model was actually tested (has data in LMS_ALL_JSON)
        SAFE_KEY=$(echo "$LMS_MODEL" | sed 's/[^a-zA-Z0-9._-]/_/g')
        if echo "$LMS_ALL_JSON" | grep -q "\"$SAFE_KEY\""; then
            M_TPS=$(echo "$LMS_ALL_JSON" | sed -n "/\"$SAFE_KEY\"/,/}/p" | sed -n 's/.*"avg_tps":[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)
            M_TTFT=$(echo "$LMS_ALL_JSON" | sed -n "/\"$SAFE_KEY\"/,/}/p" | sed -n 's/.*"avg_ttft_ms":[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
            M_LAT=$(echo "$LMS_ALL_JSON" | sed -n "/\"$SAFE_KEY\"/,/}/p" | sed -n 's/.*"avg_latency_ms":[[:space:]]*\([0-9]*\).*/\1/p' | head -1)

            SHORT_NAME=$(echo "$LMS_MODEL" | sed 's/qwen2.5-0.5b-instruct/q0.5b/; s/-mlx/mlx/')
            COL_NAMES+=("LMS $SHORT_NAME")
            COL_DECODE+=("${M_TPS:-0}")
            COL_PREFILL+=("N/A")
            COL_TTFT+=("${M_TTFT:-0}")
            COL_RT+=("${M_LAT:-0}")

            PREC_TAG="GGUF"
            echo "$LMS_MODEL" | grep -q "8bit" && PREC_TAG="MLX 8-bit"
            echo "$LMS_MODEL" | grep -q "4bit" && PREC_TAG="MLX 4-bit"
            COL_PREC+=("$PREC_TAG")
            COL_ACCEL+=("CPU/GPU")

            LMS_TESTED_NAMES+=("$LMS_MODEL")
            LMS_TESTED_TPS+=("${M_TPS:-0}")
            LMS_TESTED_TTFT+=("${M_TTFT:-0}")
            LMS_TESTED_LAT+=("${M_LAT:-0}")
        fi
    done

    NUM_COLS=${#COL_NAMES[@]}
    COL_W=16

    # Print header row
    printf "%-20s" ""
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "${COL_NAMES[$c]}"; done
    echo ""
    printf '%.0s─' $(seq 1 $((20 + NUM_COLS * COL_W))); echo ""

    # Data rows
    printf "%-20s" "Decode (t/s)"
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "${COL_DECODE[$c]}"; done
    echo ""

    printf "%-20s" "Prefill (t/s)"
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "${COL_PREFILL[$c]}"; done
    echo ""

    printf "%-20s" "TTFT (ms)"
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "${COL_TTFT[$c]}"; done
    echo ""

    printf "%-20s" "Round-trip (ms)"
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "${COL_RT[$c]}"; done
    echo ""

    printf "%-20s" "Cold start (ms)"
    printf "%${COL_W}s" "$COLD_MS"
    for c in $(seq 1 $((NUM_COLS - 1))); do printf "%${COL_W}s" "N/A"; done
    echo ""

    printf '%.0s─' $(seq 1 $((20 + NUM_COLS * COL_W))); echo ""

    printf "%-20s" "Precision"
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "${COL_PREC[$c]}"; done
    echo ""

    printf "%-20s" "Accelerator"
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "${COL_ACCEL[$c]}"; done
    echo ""

    printf "%-20s" "Timing"
    for c in $(seq 0 $((NUM_COLS - 1))); do printf "%${COL_W}s" "Wall-clock"; done
    echo ""
    echo ""

    # Append LM Studio results to JSON
    LMS_JSON_BLOCK=",
  \"lm_studio\": {
    \"port\": $LMS_PORT,
    \"models_tested\": [$(printf '"%s",' "${LMS_MODEL_LIST[@]}" | sed 's/,$//')],$( echo "$LMS_ALL_JSON" | sed '$ s/,$//' )
  }
}"
    sed -i '' '$ s/}$//' "$RESULTS_JSON"
    printf '%s\n' "$LMS_JSON_BLOCK" >> "$RESULTS_JSON"
    dim "LM Studio results added to $RESULTS_JSON"
else
    # No LM Studio -- print ANE-only comparison if we have multiple formats
    if [ "$NUM_ANE_FMTS" -gt 1 ]; then
        info "=== ANE Format Comparison ==="
        echo ""
        printf "%-20s" ""
        for fi2 in $(seq 0 $((NUM_ANE_FMTS - 1))); do printf "%16s" "ANE ${ANE_FMT_NAMES[$fi2]}"; done
        echo ""
        printf '%.0s─' $(seq 1 $((20 + NUM_ANE_FMTS * 16))); echo ""
        printf "%-20s" "Decode (t/s)"
        for fi2 in $(seq 0 $((NUM_ANE_FMTS - 1))); do printf "%16s" "${ALL_AVG_D[$fi2]}"; done
        echo ""
        printf "%-20s" "Prefill (t/s)"
        for fi2 in $(seq 0 $((NUM_ANE_FMTS - 1))); do printf "%16s" "${ALL_AVG_P[$fi2]}"; done
        echo ""
        printf "%-20s" "TTFT (ms)"
        for fi2 in $(seq 0 $((NUM_ANE_FMTS - 1))); do printf "%16s" "${ALL_AVG_TTFT[$fi2]}"; done
        echo ""
        printf "%-20s" "Round-trip (ms)"
        for fi2 in $(seq 0 $((NUM_ANE_FMTS - 1))); do printf "%16s" "${ALL_AVG_RT[$fi2]}"; done
        echo ""
        printf '%.0s─' $(seq 1 $((20 + NUM_ANE_FMTS * 16))); echo ""
        echo ""
    fi

    info "=== LM Studio Comparison ==="
    echo ""
    if [ "$LMS_REACHABLE" -eq 0 ]; then
        echo "  LM Studio server not detected on localhost:$LMS_PORT"
        echo ""
        echo "  To enable automatic comparison:"
        echo "  1. Open LM Studio, download Qwen2.5-0.5B-Instruct (GGUF + MLX variants)"
        echo "  2. Load the model, go to Developer tab > Start Server"
        echo "  3. Re-run this benchmark"
        echo ""
        echo "  Or set env vars: LMS_PORT=1234 LMS_API_KEY=your-key ./benchmark.sh"
        echo ""
        echo "  Models benchmarked by default:"
        echo "    - qwen2.5-0.5b-instruct          (GGUF)"
        echo "    - qwen2.5-0.5b-instruct-mlx@8bit (MLX 8-bit)"
        echo "    - qwen2.5-0.5b-instruct-mlx@4bit (MLX 4-bit)"
        echo ""
        echo "  Override with: LMS_MODELS='model1,model2' ./benchmark.sh"
    fi
    echo ""
    echo "  Manual test:"
    echo "  curl http://localhost:1234/api/v1/chat \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -H 'Authorization: Bearer YOUR_API_KEY' \\"
    echo "    -d '{\"model\":\"qwen2.5-0.5b-instruct\",\"system_prompt\":\"You are a helpful assistant.\",\"input\":\"What is 2+2?\"}'"
    echo ""
    echo "  ANE F16: prefill=${AVG_P} t/s, decode=${AVG_D} t/s, inference=${AVG_INF}ms"
    echo ""
    echo "  Note: LM Studio uses quantized GGUF/MLX (CPU/GPU) while we use"
    echo "  F16/Q8 weights running on CPU AMX / NEON."
fi
echo ""
