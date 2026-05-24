from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {
            "ts": time.time(),
            "event": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, int | float] = {}
        self.values: dict[str, Any] = {}

    def inc(self, key: str, amount: int | float = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def snapshot(self) -> dict[str, Any]:
        return {"counters": dict(self.counters), "values": dict(self.values)}

