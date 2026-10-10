#!/usr/bin/env python3
"""Run one preparation command and record portable wall-clock metadata."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-state", choices=("clean", "cached", "mixed", "unknown"), default="unknown"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - started
    report = {
        "operation": args.operation,
        "command": command,
        "cache_state": args.cache_state,
        "started_at_utc": started_at,
        "wall_seconds": elapsed,
        "exit_code": completed.returncode,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
