#!/usr/bin/env python3
"""
run_experiment.py — Core experiment runner for pi5-thermal-study.

Usage:
    python src/run_experiment.py --model yolov8n --format pytorch --cooling passive
    python src/run_experiment.py --model yolo11n --format onnx --cooling active --duration 5
"""

import argparse
import collections
import csv
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Repo layout ───────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR  = REPO_ROOT / "models"
DATA_DIR    = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "raw_csv"
VIDEO_PATH  = DATA_DIR / "traffic.mp4"

# Add src/ to path so logger.py is importable
sys.path.insert(0, str(REPO_ROOT / "src"))
from logger import ThermalLogger  # noqa: E402  (import after path setup)

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_MODELS  = {"yolov8n", "yolo11n", "yolo12n"}
VALID_FORMATS = {"pytorch", "onnx", "openvino"}
VALID_COOLING = {"passive", "active"}

WARMUP_FRAMES    = 200
PROGRESS_EVERY_S = 60
LOG_INTERVAL_S   = 5


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Run one thermal inference experiment.")
    p.add_argument("--model",    required=True, choices=sorted(VALID_MODELS))
    p.add_argument("--format",   required=True, choices=sorted(VALID_FORMATS))
    p.add_argument("--cooling",  required=True, choices=sorted(VALID_COOLING))
    p.add_argument("--duration", type=int, default=60,
                   help="Run duration in minutes (default: 60)")
    return p.parse_args()


def resolve_model_path(model_name: str, fmt: str) -> Path:
    if fmt == "pytorch":
        return MODELS_DIR / f"{model_name}.pt"
    elif fmt == "onnx":
        return MODELS_DIR / f"{model_name}.onnx"
    elif fmt == "openvino":
        return MODELS_DIR / f"{model_name}_openvino_model"
    raise ValueError(f"Unknown format: {fmt}")


def load_model(model_name: str, fmt: str, model_path: Path) -> YOLO:
    print(f"Loading {model_name} [{fmt}] ...")
    model = YOLO(str(model_path)) if fmt == "pytorch" else YOLO(str(model_path), task="detect")
    print("Model loaded.")
    return model


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {path.name} | {total} frames | {native:.1f} fps native")
    return cap, total


def read_frame(cap: cv2.VideoCapture):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Video unreadable after seek to frame 0.")
        return frame, True
    return frame, False


# ══════════════════════════════════════════════════════════════════════════════
# Logger thread  — no imports here, thermal_logger is passed in already built
# ══════════════════════════════════════════════════════════════════════════════

def logger_thread_fn(thermal_logger: ThermalLogger,
                     fps_state: dict,
                     fps_lock: threading.Lock,
                     stop_event: threading.Event):
    next_log = time.monotonic() + LOG_INTERVAL_S
    while not stop_event.is_set():
        if time.monotonic() >= next_log:
            with fps_lock:
                current_fps = fps_state["fps"]
            thermal_logger.log(current_fps)
            next_log += LOG_INTERVAL_S
        time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    duration_s = args.duration * 60

    # ── Validate model path ───────────────────────────────────────────────────
    model_path = resolve_model_path(args.model, args.format)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # ── Load model and video ──────────────────────────────────────────────────
    model = load_model(args.model, args.format, model_path)
    cap, video_total_frames = open_video(VIDEO_PATH)

    # ── Warm-up — before clock starts, before logger starts ──────────────────
    print(f"\nRunning {WARMUP_FRAMES}-frame warm-up (not logged) ...")
    for i in range(1, WARMUP_FRAMES + 1):
        frame, _ = read_frame(cap)
        model(frame, verbose=False)
        if i % 50 == 0:
            print(f"  [warm-up] {i}/{WARMUP_FRAMES}")
    print("Warm-up complete.\n")

    # ── Timestamp captured HERE — after warm-up, just before everything starts.
    #    Both the inference CSV and ThermalLogger use this same moment.
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stem = f"{args.model}_{args.format}_{args.cooling}_{run_ts}"

    # ── Output paths ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inference_csv_path = RESULTS_DIR / f"{run_stem}_inference.csv"
    system_csv_path    = RESULTS_DIR / f"{run_stem}_system.csv"

    # ── ThermalLogger — instantiated with fixed output path ──────────────────
    # We pass output_dir and then immediately rename so filename matches run_stem.
    # (ThermalLogger generates its own timestamp; we overwrite with ours.)
    thermal_logger = ThermalLogger(
        model_name=args.model,
        runtime_format=args.format,
        cooling_condition=args.cooling,
        output_dir=str(RESULTS_DIR)
    )
    import os
    os.rename(thermal_logger.filepath, str(system_csv_path))
    thermal_logger.filepath = str(system_csv_path)
    print(f"  → renamed to: {system_csv_path.name}")

    # ── Shared state: inference → logger thread ───────────────────────────────
    fps_lock   = threading.Lock()
    fps_state  = {"fps": 0.0}
    stop_event = threading.Event()

    # ── Start logger thread ───────────────────────────────────────────────────
    log_thread = threading.Thread(
        target=logger_thread_fn,
        args=(thermal_logger, fps_state, fps_lock, stop_event),
        daemon=True
    )
    log_thread.start()

    # ── Verify logger thread is alive before starting inference ───────────────
    time.sleep(1)
    if not log_thread.is_alive():
        print("ERROR: Logger thread died immediately. Aborting.")
        cap.release()
        sys.exit(1)

    # ── Open inference CSV ────────────────────────────────────────────────────
    inf_file   = open(inference_csv_path, "w", newline="")
    inf_writer = csv.writer(inf_file)
    inf_writer.writerow(["timestamp", "elapsed_seconds", "frame_idx", "inference_ms"])

    # ── Start clock and logger ────────────────────────────────────────────────
    thermal_logger.start()
    start_time = time.monotonic()

    # ── Print run header ──────────────────────────────────────────────────────
    print("═" * 57)
    print(f"  START : {args.model} | {args.format} | {args.cooling}")
    print(f"  Duration  : {args.duration} min")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Inference : {inference_csv_path.name}")
    print(f"  System    : {system_csv_path.name}")
    print("═" * 57)

    # ── Tracking ──────────────────────────────────────────────────────────────
    frame_idx       = 0
    loop_count      = 0
    last_progress_t = start_time
    recent_ts       = collections.deque(maxlen=300)
    all_inf_ms      = []
    elapsed         = 0.0

    # ── Inference loop ────────────────────────────────────────────────────────
    try:
        while True:
            now_mono = time.monotonic()
            elapsed  = now_mono - start_time
            if elapsed >= duration_s:
                break

            frame, looped = read_frame(cap)
            if looped:
                loop_count += 1

            t0     = time.perf_counter()
            model(frame, verbose=False)
            t1     = time.perf_counter()
            inf_ms = (t1 - t0) * 1000

            now_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            inf_writer.writerow([now_wall, round(elapsed, 2), frame_idx, round(inf_ms, 2)])
            all_inf_ms.append(inf_ms)

            recent_ts.append(t1)
            if len(recent_ts) >= 2:
                span = recent_ts[-1] - recent_ts[0]
                rolling_fps = (len(recent_ts) - 1) / span if span > 0 else 0.0
            else:
                rolling_fps = 0.0

            with fps_lock:
                fps_state["fps"] = rolling_fps

            frame_idx += 1

            if now_mono - last_progress_t >= PROGRESS_EVERY_S:
                m, s = divmod(int(elapsed), 60)
                print(f"[T+{m:02d}:{s:02d}] Frame {frame_idx:,} | "
                      f"FPS: {rolling_fps:.2f} | "
                      f"Avg inf: {sum(all_inf_ms)/len(all_inf_ms):.1f} ms")
                last_progress_t = now_mono

    except KeyboardInterrupt:
        m, s = divmod(int(elapsed), 60)
        print(f"\n⚠  Interrupted at frame {frame_idx:,} (T+{m:02d}:{s:02d})")
        print("Saving partial data ...")

    finally:
        stop_event.set()
        log_thread.join(timeout=10)
        inf_file.flush()
        inf_file.close()
        cap.release()

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "═" * 57)
        print(f"  DONE: {args.model} | {args.format} | {args.cooling}")
        print("═" * 57)
        if all_inf_ms:
            m, s = divmod(int(elapsed), 60)
            print(f"  Duration     : {m}m {s}s")
            print(f"  Total frames : {frame_idx:,}")
            print(f"  Video loops  : {loop_count}")
            print(f"  Mean FPS     : {frame_idx / elapsed:.2f}")
            print(f"  Mean inf     : {sum(all_inf_ms)/len(all_inf_ms):.1f} ms")
            print(f"  Min  inf     : {min(all_inf_ms):.1f} ms")
            print(f"  Max  inf     : {max(all_inf_ms):.1f} ms")
        print(f"  Inference CSV: {inference_csv_path.name}")
        print(f"  System CSV   : {system_csv_path.name}")
        print("═" * 57)


if __name__ == "__main__":
    main()