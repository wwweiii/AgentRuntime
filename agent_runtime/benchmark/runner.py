from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_runtime.core.loader import load_scenario
from agent_runtime.core.runtime import AgentRuntime


@dataclass(slots=True)
class ScenarioBenchmark:
    """Result of benchmarking a single scenario over multiple rounds."""

    name: str
    path: Path
    rounds: int
    results: list[dict[str, Any]] = field(default_factory=list)

    # Aggregated metrics
    avg_elapsed_s: float = 0.0
    avg_completed: int = 0
    avg_failed: int = 0
    avg_reuse_ratio: float = 0.0
    avg_sharing_ratio: float = 0.0
    avg_segments: int = 0
    avg_naive_segments: int = 0
    avg_dedup_savings: int = 0
    avg_semantic_hits: int = 0
    avg_snapshots: int = 0
    avg_throughput: float = 0.0  # tasks/s

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "rounds": self.rounds,
            "avg_elapsed_s": round(self.avg_elapsed_s, 4),
            "avg_completed": self.avg_completed,
            "avg_failed": self.avg_failed,
            "avg_reuse_ratio": round(self.avg_reuse_ratio, 4),
            "avg_sharing_ratio": round(self.avg_sharing_ratio, 4),
            "avg_segments": self.avg_segments,
            "avg_naive_segments": self.avg_naive_segments,
            "avg_dedup_savings": self.avg_dedup_savings,
            "avg_snapshots": self.avg_snapshots,
            "avg_throughput": round(self.avg_throughput, 2),
            "per_round": [
                {
                    "elapsed_s": round(r["elapsed_s"], 4),
                    "completed": r["completed"],
                    "failed": r["failed"],
                    "reuse_ratio": round(r["reuse_ratio"], 4),
                    "sharing_ratio": round(r["sharing_ratio"], 4),
                    "segments": r["segments"],
                    "naive_segments": r["naive_segments"],
                    "dedup_savings": r["dedup_savings"],
                }
                for r in self.results
            ],
        }


@dataclass(slots=True)
class ComparisonReport:
    scenarios: list[ScenarioBenchmark] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    overall_avg_reuse: float = 0.0
    overall_avg_sharing: float = 0.0
    overall_total_dedup_savings: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_elapsed_s": round(self.total_elapsed_s, 4),
            "overall_avg_reuse_ratio": round(self.overall_avg_reuse, 4),
            "overall_avg_sharing_ratio": round(self.overall_avg_sharing, 4),
            "overall_total_dedup_savings": self.overall_total_dedup_savings,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


def _estimate_naive_metrics(run_dir: Path) -> dict[str, int]:
    """Estimate what metrics would be without context deduplication.

    Without dedup, every create_segment call produces a new segment
    (no hash-hit reuse).  We reconstruct the counterfactual from the
    exported context store metrics.
    """
    contexts_path = run_dir / "contexts.json"
    if not contexts_path.exists():
        return {"naive_segments": 0, "naive_storage_tokens": 0, "dedup_savings": 0}

    data = json.loads(contexts_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    metrics = data.get("metrics", {})

    hits = (
        metrics.get("dedupe_hits", 0)
        + metrics.get("dedupe_soft_hits", 0)
        + metrics.get("dedupe_semantic_hits", 0)
    )
    misses = metrics.get("dedupe_misses", 0)
    total_calls = hits + misses

    total_tokens = sum(s.get("tokens", 0) for s in segments)
    avg_tokens = total_tokens / max(1, misses)

    naive_segments = total_calls
    naive_storage_tokens = int(naive_segments * avg_tokens)

    return {
        "naive_segments": naive_segments,
        "naive_storage_tokens": naive_storage_tokens,
        "dedup_savings": naive_storage_tokens - total_tokens,
        "semantic_hits": metrics.get("dedupe_semantic_hits", 0),
    }


class BenchmarkRunner:
    def __init__(
        self,
        scenarios: list[Path],
        rounds: int = 3,
        mock: bool = True,
        model: str | None = None,
        ollama_url: str | None = None,
    ) -> None:
        self.scenario_paths = scenarios
        self.rounds = rounds
        self.mock = mock
        self.model_override = model
        self.ollama_url_override = ollama_url

    def run(self) -> ComparisonReport:
        report = ComparisonReport()
        start_all = time.time()

        for path in self.scenario_paths:
            name = path.parent.name if path.name == "task.json" else path.stem
            sb = self._benchmark_one(name, path)
            report.scenarios.append(sb)

        report.total_elapsed_s = time.time() - start_all
        if report.scenarios:
            report.overall_avg_reuse = statistics.mean(s.avg_reuse_ratio for s in report.scenarios)
            report.overall_avg_sharing = statistics.mean(s.avg_sharing_ratio for s in report.scenarios)
            report.overall_total_dedup_savings = sum(s.avg_dedup_savings for s in report.scenarios)

        return report

    def _benchmark_one(self, name: str, path: Path) -> ScenarioBenchmark:
        sb = ScenarioBenchmark(name=name, path=path, rounds=self.rounds)

        for _ in range(self.rounds):
            config, agents, tasks, root_context = load_scenario(path)
            if self.mock:
                config.mock_model = True
            if self.model_override:
                config.model = self.model_override
                for agent in agents:
                    agent.model = self.model_override
            if self.ollama_url_override:
                config.ollama_url = self.ollama_url_override

            start = time.time()
            runtime = AgentRuntime(config, agents, tasks, root_context)
            summary = runtime.run()
            elapsed = time.time() - start

            states = summary["states"]
            ctx_metrics = summary["context"]
            naive = _estimate_naive_metrics(runtime.run_dir)

            completed = states.get("completed", 0)
            failed = states.get("failed", 0)

            sb.results.append(
                {
                    "elapsed_s": elapsed,
                    "completed": completed,
                    "failed": failed,
                    "reuse_ratio": ctx_metrics.get("reuse_ratio", 0),
                    "sharing_ratio": ctx_metrics.get("sharing_ratio", 0),
                    "segments": ctx_metrics.get("segments", 0),
                    "snapshots": ctx_metrics.get("snapshots", 0),
                    "naive_segments": naive["naive_segments"],
                    "dedup_savings": naive["dedup_savings"],
                    "semantic_hits": naive["semantic_hits"],
                    "run_id": summary["run_id"],
                }
            )

        sb.avg_elapsed_s = statistics.mean(r["elapsed_s"] for r in sb.results)
        sb.avg_completed = round(statistics.mean(r["completed"] for r in sb.results))
        sb.avg_failed = round(statistics.mean(r["failed"] for r in sb.results))
        sb.avg_reuse_ratio = statistics.mean(r["reuse_ratio"] for r in sb.results)
        sb.avg_sharing_ratio = statistics.mean(r["sharing_ratio"] for r in sb.results)
        sb.avg_segments = round(statistics.mean(r["segments"] for r in sb.results))
        sb.avg_naive_segments = round(statistics.mean(r["naive_segments"] for r in sb.results))
        sb.avg_dedup_savings = round(statistics.mean(r["dedup_savings"] for r in sb.results))
        sb.avg_semantic_hits = round(statistics.mean(r["semantic_hits"] for r in sb.results))
        sb.avg_snapshots = round(statistics.mean(r["snapshots"] for r in sb.results))
        sb.avg_throughput = sb.avg_completed / max(sb.avg_elapsed_s, 0.001)

        return sb


def run_benchmarks(
    scenarios: list[Path] | None = None,
    rounds: int = 3,
    mock: bool = True,
) -> ComparisonReport:
    if scenarios is None:
        examples_dir = Path(__file__).resolve().parents[2] / "examples"
        scenarios = sorted(examples_dir.glob("*/task.json"))
    runner = BenchmarkRunner(scenarios=scenarios, rounds=rounds, mock=mock)
    return runner.run()
