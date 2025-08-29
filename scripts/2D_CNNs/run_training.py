#!/usr/bin/env python3
"""
Run multiple TensorFlow training scripts sequentially.
- Project structure:
    /home/lennart/work/
        logs/
        tfrs/
        Brain_Mets_Classification/   <-- repo(s); a 'scripts' folder exists several levels deep here
- Usage (master log captured by shell redirection):
    nohup python3 run_training.py > training_output.log &
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# =========================
# EDIT THESE TWO CONSTANTS:
# =========================
PROJECT_BASE_DIR = Path("/home/lennart/work").expanduser().resolve()

# If you KNOW the exact scripts directory, set it here, e.g.:
# SCRIPTS_DIR = PROJECT_BASE_DIR / "Brain_Mets_Classification" / "my-repo" / "experiments" / "scr ipts"
# Otherwise leave as None and the script will auto-discover by searching under PROJECT_BASE_DIR/Brain_Mets_Classification and PROJECT_BASE_DIR
SCRIPTS_DIR: Optional[Path] = None

# -------------------------
# Jobs to run (toggle here)
# -------------------------
JOBS: List[str] = [
    # "2D_CNN_conv.py",
    # "2D_CNN_resnet34.py",
    # "2D_CNN_resnet152.py",
    # "2D_CNN_resnext50.py",
    # "2D_CNN_resnext101.py",
    #"2D_CNN_transfer_bit.py",
    #"2D_CNN_transfer_efficientv2.py",
    "2D_CNN_transfer_inceptionv3.py",
    "2D_CNN_transfer_resnet50v2.py",
    "2D_transfer_vit.py",
]

# =========================
# Implementation
# =========================

def now_str(fmt="%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(fmt)

def discover_script_path(name: str, cache: Dict[str, Optional[Path]]) -> Optional[Path]:
    """
    Find the absolute path to 'name' by searching likely locations.
    Caches results to avoid repeated filesystem walks.
    """
    if name in cache:
        return cache[name]

    # 1) If a fixed SCRIPTS_DIR is set, try there first
    if SCRIPTS_DIR:
        candidate = SCRIPTS_DIR / name
        if candidate.exists():
            cache[name] = candidate
            return candidate

    # 2) Search under Brain_Mets_Classification/ first, then the entire project base as fallback
    search_roots = []
    github_dir = PROJECT_BASE_DIR / "Brain_Mets_Classification"
    if github_dir.exists():
        search_roots.append(github_dir)
    search_roots.append(PROJECT_BASE_DIR)

    for root in search_roots:
        # rglob is fine here; scripts are a limited set of files
        for p in root.rglob(name):
            if p.is_file():
                cache[name] = p.resolve()
                return cache[name]

    cache[name] = None
    return None

def run_script(script_path: Path, logs_dir: Path, env: dict) -> int:
    script_name = script_path.name
    print(f'--- Starting training for: {script_name} ---', flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{script_name.replace('.py','')}_{timestamp}.log"

    with log_file.open("wb") as lf:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            env=env,
            stdout=lf,
            stderr=lf,
            check=False,  # keep going on failure
        )
        code = proc.returncode

    if code == 0:
        print(f'--- Finished training for: {script_name} ---', flush=True)
    else:
        print(f'--- FAILED training for: {script_name} (exit {code}) ---', flush=True)
        print(f'    See log: {log_file}', flush=True)

    return code

def main() -> int:
    # Validate base directories
    if not PROJECT_BASE_DIR.exists():
        print(f"ERROR: PROJECT_BASE_DIR does not exist: {PROJECT_BASE_DIR}", file=sys.stderr, flush=True)
        return 2

    logs_dir = PROJECT_BASE_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Propagate PROJECT_BASE_DIR & TFRS for your training scripts
    env = os.environ.copy()
    env["PROJECT_BASE_DIR"] = str(PROJECT_BASE_DIR)
    # Keep your original tfrs location available if your scripts expect it
    env["TFRS_DIR"] = str((PROJECT_BASE_DIR / "tfrs").resolve())

    print("=" * 70, flush=True)
    print(f"Starting run at {now_str()}", flush=True)
    print("=" * 70, flush=True)

    # Resolve each job to an absolute path
    path_cache: Dict[str, Optional[Path]] = {}
    results = []  # (name, exit_code, resolved_path or None)

    for name in JOBS:
        script_path = discover_script_path(name, path_cache)
        if script_path is None:
            print(f"ERROR: Could not find script '{name}' under {PROJECT_BASE_DIR}/Brain_Mets_Classification or {PROJECT_BASE_DIR}", file=sys.stderr, flush=True)
            results.append((name, 127, None))  # 127 ~ command not found
            continue

        code = run_script(script_path, logs_dir, env)
        results.append((name, code, script_path))

    print("=" * 70, flush=True)
    print(f"All runs finished at {now_str()}", flush=True)
    print("=" * 70, flush=True)

    # Summary
    if results:
        width = max(len(n) for n, _, _ in results)
        print("Summary:", flush=True)
        for n, code, _ in results:
            status = "OK" if code == 0 else f"FAIL ({code})"
            print(f"  {n.ljust(width)} : {status}", flush=True)

    # Exit non-zero if any failed
    any_fail = any(code != 0 for _, code, _ in results)
    worst = max((code for _, code, _ in results), default=0)
    return 0 if not any_fail else (worst if worst != 0 else 1)

if __name__ == "__main__":
    sys.exit(main())
