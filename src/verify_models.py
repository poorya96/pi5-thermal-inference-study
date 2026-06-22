#!/usr/bin/env python3
"""
verify_models.py — Pre-flight check for all 9 model/format combos.
Loads each model, runs one real frame through it, prints a summary table.

Usage:
    python src/verify_models.py
    python src/verify_models.py --verbose
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
VIDEO_PATH = REPO_ROOT / "data" / "traffic.mp4"

# ── 9 combos: (display_model, display_format, path_to_pass_to_YOLO) ─────────
COMBOS = [
    ("yolov8n",  "pytorch",   MODELS_DIR / "yolov8n.pt"),
    ("yolov8n",  "onnx",      MODELS_DIR / "yolov8n.onnx"),
    ("yolov8n",  "openvino",  MODELS_DIR / "yolov8n_openvino_model"),
    ("yolo11n",  "pytorch",   MODELS_DIR / "yolo11n.pt"),
    ("yolo11n",  "onnx",      MODELS_DIR / "yolo11n.onnx"),
    ("yolo11n",  "openvino",  MODELS_DIR / "yolo11n_openvino_model"),
    ("yolo12n",  "pytorch",   MODELS_DIR / "yolo12n.pt"),
    ("yolo12n",  "onnx",      MODELS_DIR / "yolo12n.onnx"),
    ("yolo12n",  "openvino",  MODELS_DIR / "yolo12n_openvino_model"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def read_one_frame(video_path: Path) -> any:
    """Read the 10th frame from the video (avoids black first frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame = None
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Video has fewer than 10 frames.")
    cap.release()
    return frame


def test_combo(model_name, fmt, model_path, frame, verbose):
    """Load one model, run one frame, return (passed, inference_ms, error_str)."""
    if verbose:
        print(f"  Loading {model_name} [{fmt}] from {model_path} ...")
    try:
        t0 = time.perf_counter()
        model = YOLO(str(model_path))
        results = model(frame, verbose=False)
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000

        # Sanity-check: result object must exist and have boxes attribute
        if results is None or len(results) == 0:
            raise ValueError("model() returned empty results")
        _ = results[0].boxes  # raises if attribute missing

        del model
        gc.collect()

        if verbose:
            print(f"  ✓ Done — {inference_ms:.0f} ms")
        return True, inference_ms, ""

    except Exception as e:
        gc.collect()
        if verbose:
            print(f"  ✗ FAILED — {e}")
        return False, 0.0, str(e)


def print_table(results):
    """Print a formatted summary table."""
    col_w = [10, 10, 10, 18, 0]  # last col is dynamic
    header = f"{'Model':<{col_w[0]}}  {'Format':<{col_w[1]}}  {'Status':<{col_w[2]}}  {'Inference (ms)':<{col_w[3]}}"
    sep    = "─" * len(header)

    print()
    print("  MODEL VERIFICATION RESULTS")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for model_name, fmt, passed, ms, err in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        ms_str = f"{ms:.0f}" if passed else "—"
        print(f"  {model_name:<{col_w[0]}}  {fmt:<{col_w[1]}}  {status:<{col_w[2]}}  {ms_str:<{col_w[3]}}")
        if not passed:
            print(f"  {'':>{col_w[0]}}  {'':>{col_w[1]}}  Error: {err}")
    print(f"  {sep}")

    total  = len(results)
    passed = sum(1 for *_, p, _, __ in results if p)
    print(f"  {passed}/{total} passed")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verify all 9 model/format combos.")
    parser.add_argument("--verbose", action="store_true", help="Print per-step details")
    args = parser.parse_args()

    # 1. Read one frame
    print(f"\nReading test frame from {VIDEO_PATH} ...")
    try:
        frame = read_one_frame(VIDEO_PATH)
        print(f"Frame shape: {frame.shape}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # 2. Test all 9 combos
    results = []
    total = len(COMBOS)
    for i, (model_name, fmt, model_path) in enumerate(COMBOS, 1):
        print(f"\n[{i}/{total}] Testing {model_name} | {fmt} ...")
        passed, ms, err = test_combo(model_name, fmt, model_path, frame, args.verbose)
        results.append((model_name, fmt, passed, ms, err))

    # 3. Print summary table
    print_table(results)

    # 4. Exit code
    all_passed = all(r[2] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()