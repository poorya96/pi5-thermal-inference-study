#!/usr/bin/env bash
# run_all.sh — Runs all 18 thermal inference experiments in order.
# Usage: bash run_all.sh
# Run inside tmux: tmux new -s thermal

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/thermal_study"
RUNNER="$REPO_DIR/src/run_experiment.py"
LOG_FILE="$REPO_DIR/results/run_log.txt"
RESULTS_DIR="$REPO_DIR/results/raw_csv"

mkdir -p "$RESULTS_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

get_temp() {
    vcgencmd measure_temp | grep -oP '\d+\.\d+'
}

wait_for_cooldown() {
    local target=45.0
    log "COOLDOWN START — waiting for temp < ${target}°C"
    while true; do
        local temp
        temp=$(get_temp)
        echo "  [$(date '+%H:%M:%S')] Temp: ${temp}°C"
        if (( $(echo "$temp < $target" | bc -l) )); then
            log "COOLDOWN DONE — temp at ${temp}°C — waiting 60s for stability"
            sleep 60
            break
        fi
        sleep 30
    done
}

run_experiment() {
    local run_num=$1
    local total=$2
    local model=$3
    local format=$4
    local cooling=$5

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  RUN ${run_num}/${total}: ${model} | ${format} | ${cooling}"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  ⚡ NOTE: Record power meter at T+1min and T+45min"
    echo "════════════════════════════════════════════════════"

    log "RUN ${run_num}/${total} START: ${model} ${format} ${cooling}"

    set +e
    source "$VENV_DIR/bin/activate"
    python "$RUNNER" \
        --model "$model" \
        --format "$format" \
        --cooling "$cooling" \
        --duration 60
    local exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        log "RUN ${run_num}/${total} DONE: ${model} ${format} ${cooling} — exit code 0 ✓"
    else
        log "RUN ${run_num}/${total} FAILED: ${model} ${format} ${cooling} — exit code ${exit_code} ✗"
        echo ""
        echo "  ⚠ WARNING: Run ${run_num} exited with code ${exit_code}."
        echo "  Continuing to next run. Re-run this experiment manually later."
        echo ""
    fi
}

# ── Startup warning ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  pi5-thermal-study — Full 18-Run Session"
echo "  Log: $LOG_FILE"
echo ""
echo "  ⚠ WARNING: Run this inside tmux."
echo "  SSH disconnect will kill the session."
echo "  If not in tmux: Ctrl+C now, then:"
echo "    tmux new -s thermal && bash run_all.sh"
echo "════════════════════════════════════════════════════"
echo ""
read -rp "  Press Enter to start the passive cooling block (runs 1–9)..."

log "SESSION START — 18-run thermal study"

# ── Initial temp check ────────────────────────────────────────────────────────
TEMP_NOW=$(get_temp)
echo ""
echo "  Current temp: ${TEMP_NOW}°C"
if (( $(echo "$TEMP_NOW > 45.0" | bc -l) )); then
    echo "  Pi is warm. Running cooldown before first experiment."
    wait_for_cooldown
else
    echo "  ✓ Temp OK — starting in 5 seconds..."
    sleep 5
fi

# ══════════════════════════════════════════════════════
# PASSIVE COOLING BLOCK (runs 1–9)
# ══════════════════════════════════════════════════════
run_experiment  1 18 yolov8n  pytorch  passive; wait_for_cooldown
run_experiment  2 18 yolov8n  onnx     passive; wait_for_cooldown
run_experiment  3 18 yolov8n  openvino passive; wait_for_cooldown
run_experiment  4 18 yolo11n  pytorch  passive; wait_for_cooldown
run_experiment  5 18 yolo11n  onnx     passive; wait_for_cooldown
run_experiment  6 18 yolo11n  openvino passive; wait_for_cooldown
run_experiment  7 18 yolo12n  pytorch  passive; wait_for_cooldown
run_experiment  8 18 yolo12n  onnx     passive; wait_for_cooldown
run_experiment  9 18 yolo12n  openvino passive

# ── Pause for cooler attachment ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  PASSIVE BLOCK COMPLETE (9/18 runs done)"
echo ""
echo "  ACTION REQUIRED:"
echo "  1. Attach the active cooler to the Pi now"
echo "  2. Verify the fan is spinning"
echo "  3. Press Enter to continue with the active block"
echo "════════════════════════════════════════════════════"
read -rp "  Ready? Press Enter to start active cooling block (runs 10–18)..."

log "ACTIVE BLOCK START — cooler attached by user"
wait_for_cooldown

# ══════════════════════════════════════════════════════
# ACTIVE COOLING BLOCK (runs 10–18)
# ══════════════════════════════════════════════════════
run_experiment 10 18 yolov8n  pytorch  active; wait_for_cooldown
run_experiment 11 18 yolov8n  onnx     active; wait_for_cooldown
run_experiment 12 18 yolov8n  openvino active; wait_for_cooldown
run_experiment 13 18 yolo11n  pytorch  active; wait_for_cooldown
run_experiment 14 18 yolo11n  onnx     active; wait_for_cooldown
run_experiment 15 18 yolo11n  openvino active; wait_for_cooldown
run_experiment 16 18 yolo12n  pytorch  active; wait_for_cooldown
run_experiment 17 18 yolo12n  onnx     active; wait_for_cooldown
run_experiment 18 18 yolo12n  openvino active

# ── Session complete ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  ALL 18 RUNS COMPLETE"
echo "  Session ended: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: $RESULTS_DIR"
echo "  Log:     $LOG_FILE"
echo "════════════════════════════════════════════════════"
log "SESSION COMPLETE — all 18 runs finished"
