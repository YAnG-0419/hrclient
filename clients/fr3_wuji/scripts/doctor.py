#!/usr/bin/env python3
"""Read-only environment audit for the FR3/Wuji EgoAV78 client."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def module_state(name):
    return "ok" if importlib.util.find_spec(name) is not None else "missing"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05-root", type=Path, default=Path("/home/user/lpy/Pi05"))
    parser.add_argument("--replay-root", type=Path, default=Path("/home/user/lpy/fr3_wuji_eef_replay"))
    parser.add_argument("--calibration", type=Path,
                        default=Path("/home/user/tomato_eef/conversion/calibration.yaml"))
    args = parser.parse_args(argv)
    root = args.pi05_root.expanduser().resolve()
    replay = args.replay_root.expanduser().resolve()
    required = [
        "deploy/fr3_wuji/observation.py",
        "deploy/fr3_wuji/observe.py",
        ".deployment/oneshot-owner.lock",
    ]
    report = {
        "python": sys.executable,
        "python_version": list(sys.version_info[:3]),
        "ssh": shutil.which("ssh"),
        "mixed76_modules": {name: module_state(name) for name in ("numpy", "scipy", "PIL")},
        "egoav78_modules": {name: module_state(name) for name in ("numpy", "scipy", "yaml")},
        "pi05_root": str(root),
        "pi05_files": {relative: (root / relative).is_file() for relative in required},
        "replay_root": str(replay),
        "replay_files": {relative: (replay / relative).is_file() for relative in ("core.py", "limits.yaml")},
        "calibration": str(args.calibration.expanduser().resolve()),
        "calibration_exists": args.calibration.expanduser().is_file(),
    }
    report["mixed76_ready"] = bool(report["ssh"]) and all(
        state == "ok" for state in report["mixed76_modules"].values()
    )
    report["egoav78_ready"] = report["mixed76_ready"] and all(
        state == "ok" for state in report["egoav78_modules"].values()
    ) and all(report["pi05_files"].values()) and all(report["replay_files"].values()) \
        and report["calibration_exists"]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["egoav78_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
