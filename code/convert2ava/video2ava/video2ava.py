#!/usr/bin/env python3
"""
converter_full.py — single‑file mp4 → AVA‑frames pipeline
=========================================================

Key features
------------
* **Single log file** – all stages write to `LOG_DIR / LOG_FILE` when logging is
  enabled.
* **No intermediate folder** – frames are written directly into
  `FRAMES_DIR/<clip_name>/`.
* **Config‑level switch `ENABLE_LOG`** – set `True`/`False` at the top of the
  file to enable or disable logging globally; the command‑line flag `--log`
  still exists and will *override* this default.

Usage
-----
```bash
python converter_full.py              # obeys ENABLE_LOG default
python converter_full.py --log        # force logging, regardless of default
```
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import ffmpeg

# =============================================================================
# Configuration variables
# =============================================================================
INPUT_DIR: Path = Path("/Users/leiqi/Desktop/github/videos")            # root directory containing .mp4 files (searched recursively)
FRAMES_DIR: Path = Path("/Users/leiqi/Desktop/github/frames")      # final AVA‑style frame output directory

LOG_DIR: Path = Path("../logs")               # directory where the log file will be stored
LOG_FILE = "converter.log"                     # name of the single log file

FPS: int = 30                                  # frame‑extraction rate (frames per second)

ENABLE_LOG: bool = False                       # True to write the log file by default

# =============================================================================
# Logging helpers
# =============================================================================

def setup_logging():
    """Redirect `print()` to `logging.info` and write to both file and stdout."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / LOG_FILE, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Make `print()` a synonym for `logging.info` so the rest of the code uses it transparently
    global print  # noqa: PLW0603 – intentional override of the built‑in
    print = logging.info  # type: ignore

# =============================================================================
# mp4 → jpg (frames are written directly into FRAMES_DIR)
# =============================================================================

def _clear_folder(folder: Path):
    """Delete *folder* if it exists and recreate it empty."""
    if folder.exists():
        print("Clearing existing output folder\n")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def _extract_frames(input_root: Path, output_root: Path, fps: int):
    """Extract *fps* frames from every .mp4 in *input_root* into *output_root*."""
    videos = list(input_root.rglob("*.mp4"))
    total = len(videos)

    for i, video in enumerate(videos, 1):
        clip_name = video.stem
        out_dir = output_root / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pattern = out_dir / f"{clip_name}_%06d.jpg"

        print(f"[{i}/{total}] processing {clip_name}")
        try:
            (
                ffmpeg
                .input(str(video))
                .output(str(out_pattern), r=fps, qscale=1)
                .run(overwrite_output=True, quiet=True)
            )
            print("OK\n")
        except ffmpeg.Error as exc:
            stderr = getattr(exc, "stderr", b"").decode(errors="ignore")
            print(f"[ERROR] Failed on {clip_name}\n{stderr}\n")


def mp4tojpg_run():
    """Run the extraction stage and report its duration."""
    print("Start Program: mp4tojpg ...\n")
    start = time.time()

    _clear_folder(FRAMES_DIR)
    _extract_frames(INPUT_DIR, FRAMES_DIR, FPS)

    _print_elapsed(time.time() - start)

# =============================================================================
# Orchestrator
# =============================================================================

def _print_elapsed(elapsed: float):
    if elapsed < 60:
        print(f"\nProgram finished. Total time: {elapsed:.2f} seconds.\n")
    else:
        m, s = divmod(elapsed, 60)
        print(f"\nProgram finished. Total time: {int(m)} minutes {s:.2f} seconds.\n")


def run(*, logfile: bool):
    """Entry point for the whole pipeline."""
    if logfile:
        setup_logging()

    start = time.time()
    print("Start converter...\n")

    mp4tojpg_run()

    elapsed = time.time() - start
    if elapsed < 60:
        t = f"{elapsed:.2f} seconds"
    else:
        m, s = divmod(elapsed, 60)
        t = f"{int(m)} minutes {s:.2f} seconds"
    print(f"Converter finished. Total time: {t}. Whole program ends.")

# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="mp4 ➜ AVA‑style frames — single log file",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Force writing logs to logs/converter.log in addition to the console.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Command‑line flag overrides default switch
    effective_log = args.log or ENABLE_LOG
    run(logfile=effective_log)
