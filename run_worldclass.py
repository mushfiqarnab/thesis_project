#!/usr/bin/env python3
"""One-command bootstrap/runner for the world-class diffusion engine.

Usage:
  python run_worldclass.py --pilot
  python run_worldclass.py --full
  python run_worldclass.py --full -- --num-variants 2 --use-lpips

The wrapper creates an isolated .venv_worldclass, installs dependencies, verifies
imports, then executes commercial_diffusion_engine_worldclass.py. It intentionally
never weakens the engine's acceptance gates; CPU is supported as a last-resort
execution mode, but remains much slower than CUDA.
"""
from __future__ import annotations
import argparse, os, platform, subprocess, sys, venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "commercial_diffusion_engine_worldclass.py"
VENV = ROOT / ".venv_worldclass"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PIP = VENV / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")

BASE = [
    "diffusers>=0.40,<0.41", "transformers>=4.45", "accelerate>=1.0",
    "safetensors", "numpy", "pandas", "pillow", "scikit-image", "tqdm",
    "opencv-python", "insightface", "onnxruntime", "lpips",
]

def run(cmd, **kw):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)

def ensure_venv():
    if not PY.exists():
        print(f"[BOOT] Creating isolated environment: {VENV}")
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(VENV)
    run([str(PY), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

def install_deps():
    # Torch is installed separately so pip can resolve its platform wheel.
    run([str(PY), "-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"])
    run([str(PY), "-m", "pip", "install", "--upgrade", *BASE])
    # GPU ORT is optional. If it cannot be installed, CPU ORT remains valid.
    try:
        run([str(PY), "-m", "pip", "install", "--upgrade", "onnxruntime-gpu"])
    except subprocess.CalledProcessError:
        print("[WARN] onnxruntime-gpu installation failed; keeping CPU onnxruntime.")

def verify():
    code = r'''
import torch
print("[CHECK] torch", torch.__version__, "CUDA=", torch.cuda.is_available())
import diffusers, transformers, accelerate, insightface, cv2, pandas, skimage, lpips
print("[CHECK] diffusers", diffusers.__version__)
print("[CHECK] insightface", getattr(insightface, "__version__", "unknown"))
print("[CHECK] all required imports OK")
'''
    run([str(PY), "-c", code])

def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="run 5 samples")
    mode.add_argument("--full", action="store_true", help="run the complete configured dataset")
    ap.add_argument("--no-install", action="store_true", help="do not install/update packages")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--source-csv", default="data/csv/approved_pristine_manifest.csv")
    ap.add_argument("--allow-missing-split", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("engine_args", nargs=argparse.REMAINDER, help="arguments passed verbatim to engine after --")
    args = ap.parse_args()

    if not ENGINE.exists():
        raise SystemExit(f"Engine not found: {ENGINE}")
    ensure_venv()
    if not args.no_install:
        install_deps()
    verify()

    cmd = [str(PY), str(ENGINE), "--project-root", str(ROOT), "--source-csv", args.source_csv]
    if args.pilot:
        cmd += ["--max-samples", "5"]
    elif args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.allow_missing_split:
        cmd.append("--allow-missing-split")
    if not args.no_lpips:
        cmd.append("--use-lpips")
    if args.engine_args:
        extra = args.engine_args[1:] if args.engine_args and args.engine_args[0] == "--" else args.engine_args
        cmd += extra
    print("\n[RUN] Starting research engine. Acceptance gates remain enabled.")
    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
