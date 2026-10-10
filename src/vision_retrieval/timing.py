"""Small, portable timing-report writer for offline experiment steps."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from typing import Mapping, Optional


def write_timing_report(
    path: Path,
    operation: str,
    total_seconds: float,
    steps: Mapping[str, float],
    details: Optional[dict] = None,
) -> dict:
    durations = {name: float(value) for name, value in steps.items()}
    values = [float(total_seconds), *durations.values()]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("timing durations must be finite and non-negative")

    report = {
        "operation": operation,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": float(total_seconds),
        "steps_seconds": durations,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "details": details or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return report
