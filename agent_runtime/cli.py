from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_runtime.benchmark.runner import BenchmarkRunner, ComparisonReport
from agent_runtime.core.loader import load_scenario
from agent_runtime.core.runtime import AgentRuntime


def main() -> None:
    parser = argparse.ArgumentParser(prog="arun", description="Agent Runtime CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    submit = sub.add_parser("submit", help="submit a multi-agent scenario")
    submit.add_argument("scenario", type=Path)
    submit.add_argument("--mock", action="store_true", help="use deterministic mock model instead of Ollama")
    submit.add_argument("--model", default=None)
    submit.add_argument("--ollama-url", default=None)
    submit.add_argument("--max-workers", type=int, default=None)
    submit.add_argument("--max-parallel-model-calls", type=int, default=None)
    submit.add_argument("--timeout", type=int, default=None, help="override all task timeouts (seconds)")
    submit.add_argument("--semantic-threshold", type=float, default=None, help="cosine similarity threshold for semantic dedup (0.0-1.0)")

    inspect = sub.add_parser("inspect", help="inspect run directories")
    inspect.add_argument("path", type=Path, nargs="?", default=Path("runs"))

    bench = sub.add_parser("benchmark", help="run performance comparison across all scenarios")
    bench.add_argument("--mock", action="store_true", default=True)
    bench.add_argument("--no-mock", dest="mock", action="store_false", help="use real Ollama model instead of mock")
    bench.add_argument("--scenario", type=Path, default=None, help="run a single scenario instead of all")
    bench.add_argument("--rounds", type=int, default=3, help="number of rounds per scenario (default: 3)")
    bench.add_argument("--model", default=None)
    bench.add_argument("--ollama-url", default=None)
    bench.add_argument("--json", dest="json_output", action="store_true", help="output full JSON report to stdout")

    serve = sub.add_parser("serve", help="serve local dashboard files")
    serve.add_argument("--port", type=int, default=8765)

    experiment = sub.add_parser("experiment", help="run A/B comparison: context reuse vs none, context-aware vs random scheduling")
    experiment.add_argument("--mock", action="store_true", default=True)
    experiment.add_argument("--no-mock", dest="mock", action="store_false", help="use real Ollama model")
    experiment.add_argument("--scenario", type=Path, default=None, help="single scenario; omit to run all")
    experiment.add_argument("--rounds", type=int, default=3, help="rounds per configuration (default: 3)")
    experiment.add_argument("--model", default=None)
    experiment.add_argument("--ollama-url", default=None)
    experiment.add_argument("--json", dest="json_output", action="store_true", help="output JSON report")

    args = parser.parse_args()
    if args.cmd == "submit":
        _submit(args)
    elif args.cmd == "inspect":
        _inspect(args.path)
    elif args.cmd == "benchmark":
        _benchmark(args)
    elif args.cmd == "serve":
        _serve(args.port)
    elif args.cmd == "experiment":
        _experiment(args)


def _submit(args: argparse.Namespace) -> None:
    config, agents, tasks, root_context = load_scenario(args.scenario)
    if args.mock:
        config.mock_model = True
    if args.model:
        config.model = args.model
        for agent in agents:
            agent.model = args.model
    if args.ollama_url:
        config.ollama_url = args.ollama_url
    if args.max_workers:
        config.max_workers = args.max_workers
    if args.max_parallel_model_calls:
        config.max_parallel_model_calls = args.max_parallel_model_calls
    if args.timeout:
        for task in tasks:
            task.failure_policy.timeout_s = args.timeout
    if args.semantic_threshold is not None:
        config.semantic_threshold = args.semantic_threshold

    runtime = AgentRuntime(config, agents, tasks, root_context)
    summary = runtime.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _inspect(path: Path) -> None:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if not path.exists():
        print(f"No runs found at {path}")
        return
    for run_dir in sorted([item for item in path.iterdir() if item.is_dir()]):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"{run_dir.name}: no metrics.json")
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        states = data.get("states", {})
        context = data.get("context", {})
        print(
            f"{run_dir.name}: completed={states.get('completed', 0)} "
            f"failed={states.get('failed', 0)} reuse={context.get('reuse_ratio', 0):.2f} "
            f"dir={run_dir}"
        )


def _benchmark(args: argparse.Namespace) -> None:
    if args.scenario:
        paths = [args.scenario]
    else:
        examples_dir = Path(__file__).resolve().parent.parent / "examples"
        paths = sorted(examples_dir.glob("*/task.json"))

    if not paths:
        print("No scenarios found.")
        return

    runner = BenchmarkRunner(
        scenarios=paths,
        rounds=args.rounds,
        mock=args.mock,
        model=args.model,
        ollama_url=args.ollama_url,
    )
    report = runner.run()

    if args.json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    _print_report(report)


def _print_report(report: ComparisonReport) -> None:
    sep = "=" * 82
    print(f"\n{sep}")
    print("  Agent Runtime — Performance Comparison Report")
    print(f"{sep}")
    header = f"  {'Scenario':<28s} {'Tasks':>6s} {'Time(s)':>8s} {'Thruput':>8s} {'Reuse%':>7s} {'Sharing%':>8s} {'SegSave%':>8s}"
    print(header)
    print(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    for sb in report.scenarios:
        total_tasks = sb.avg_completed + sb.avg_failed
        seg_save = _pct(sb.avg_naive_segments - sb.avg_segments, sb.avg_naive_segments)
        print(
            f"  {sb.name:<28s} {total_tasks:>6d} {sb.avg_elapsed_s:>8.3f} "
            f"{sb.avg_throughput:>7.2f}/s {sb.avg_reuse_ratio*100:>6.1f}% {sb.avg_sharing_ratio*100:>7.1f}% {seg_save:>7.1f}%"
        )

    print(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    print(
        f"  {'OVERALL (avg)':<28s} {'':>6s} {report.total_elapsed_s:>8.3f} "
        f"{'':>8s} {report.overall_avg_reuse*100:>6.1f}% {report.overall_avg_sharing*100:>7.1f}%"
    )
    print(f"{sep}")

    print("\n  Per-Scenario Detail:")
    for sb in report.scenarios:
        print(f"\n  ── {sb.name}  ({sb.rounds} rounds) ──")
        print(f"     context segments:      {sb.avg_segments} (naive would be {sb.avg_naive_segments})")
        print(f"     dedup segment savings: {sb.avg_dedup_savings} token-entries avoided")
        print(f"     semantic hits:         {sb.avg_semantic_hits}")
        print(f"     reuse ratio:           {sb.avg_reuse_ratio:.3f}  |  sharing ratio: {sb.avg_sharing_ratio:.3f}")
        print(f"     snapshots:             {sb.avg_snapshots}")
        print(f"     throughput:            {sb.avg_throughput:.2f} tasks/s")
        print(f"     avg task completed:    {sb.avg_completed}  |  avg failed: {sb.avg_failed}")

    print(f"\n  Report saved under runs/*/benchmark.json")
    print(f"  Total benchmark elapsed: {report.total_elapsed_s:.3f}s\n")


def _pct(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return max(0.0, part / whole * 100)


@dataclass(slots=True)
class ExperimentResult:
    name: str
    elapsed_s: float
    completed: int
    failed: int
    segments: int
    snapshots: int
    reuse_ratio: float
    sharing_ratio: float
    total_tokens_est: int


@dataclass(slots=True)
class ExperimentComparison:
    scenario: str
    rounds: int
    baseline: ExperimentResult | None = None
    no_reuse: ExperimentResult | None = None
    random_sched: ExperimentResult | None = None

    def improvements(self) -> dict[str, Any]:
        if self.baseline is None:
            return {}
        result: dict[str, Any] = {}
        if self.no_reuse is not None:
            result["reuse_segment_savings_pct"] = _pct(
                self.no_reuse.segments - self.baseline.segments, self.no_reuse.segments,
            )
            result["reuse_time_savings_pct"] = _pct(
                self.no_reuse.elapsed_s - self.baseline.elapsed_s, self.no_reuse.elapsed_s,
            )
            result["reuse_token_savings_pct"] = _pct(
                self.no_reuse.total_tokens_est - self.baseline.total_tokens_est,
                self.no_reuse.total_tokens_est,
            )
        if self.random_sched is not None:
            result["sched_completion_pct"] = _pct(
                self.baseline.completed - self.random_sched.completed, max(1, self.baseline.completed),
            )
            result["sched_time_savings_pct"] = _pct(
                self.random_sched.elapsed_s - self.baseline.elapsed_s, self.random_sched.elapsed_s,
            )
        return result


def _run_experiment_round(path: Path, mock: bool, model: str | None, ollama_url: str | None, disable_reuse: bool, random_sched: bool) -> ExperimentResult:
    config, agents, tasks, root_context = load_scenario(path)
    if mock:
        config.mock_model = True
    if model:
        config.model = model
        for agent in agents:
            agent.model = model
    if ollama_url:
        config.ollama_url = ollama_url
    config.disable_context_reuse = disable_reuse
    config.random_scheduling = random_sched

    start = time.time()
    runtime = AgentRuntime(config, agents, tasks, root_context)
    summary = runtime.run()
    elapsed = time.time() - start

    states = summary["states"]
    ctx = summary["context"]
    counters = summary.get("metrics", {}).get("counters", {})

    return ExperimentResult(
        name=path.parent.name if path.name == "task.json" else path.stem,
        elapsed_s=elapsed,
        completed=states.get("completed", 0),
        failed=states.get("failed", 0),
        segments=ctx.get("segments", 0),
        snapshots=ctx.get("snapshots", 0),
        reuse_ratio=ctx.get("reuse_ratio", 0),
        sharing_ratio=ctx.get("sharing_ratio", 0),
        total_tokens_est=counters.get("model.prompt_tokens_est", 0)
        + counters.get("model.output_tokens_est", 0)
        + counters.get("model.context_tokens_est", 0),
    )


def _avg_results(results: list[ExperimentResult]) -> ExperimentResult | None:
    if not results:
        return None
    return ExperimentResult(
        name=results[0].name,
        elapsed_s=statistics.mean(r.elapsed_s for r in results),
        completed=round(statistics.mean(r.completed for r in results)),
        failed=round(statistics.mean(r.failed for r in results)),
        segments=round(statistics.mean(r.segments for r in results)),
        snapshots=round(statistics.mean(r.snapshots for r in results)),
        reuse_ratio=statistics.mean(r.reuse_ratio for r in results),
        sharing_ratio=statistics.mean(r.sharing_ratio for r in results),
        total_tokens_est=round(statistics.mean(r.total_tokens_est for r in results)),
    )


def _experiment(args: argparse.Namespace) -> None:
    if args.scenario:
        paths = [args.scenario]
    else:
        examples_dir = Path(__file__).resolve().parent.parent / "examples"
        paths = sorted(examples_dir.glob("*/task.json"))

    if not paths:
        print("No scenarios found.")
        return

    comparisons: list[ExperimentComparison] = []
    total_start = time.time()

    for path in paths:
        name = path.parent.name if path.name == "task.json" else path.stem
        comp = ExperimentComparison(scenario=name, rounds=args.rounds)
        model = args.model
        ollama = args.ollama_url

        # ── Baseline: context reuse ON, context-aware scheduling ──
        baseline_results: list[ExperimentResult] = []
        for _ in range(args.rounds):
            baseline_results.append(_run_experiment_round(path, args.mock, model, ollama, disable_reuse=False, random_sched=False))
        comp.baseline = _avg_results(baseline_results)

        # ── Configuration A: context reuse OFF ──
        no_reuse_results: list[ExperimentResult] = []
        for _ in range(args.rounds):
            no_reuse_results.append(_run_experiment_round(path, args.mock, model, ollama, disable_reuse=True, random_sched=False))
        comp.no_reuse = _avg_results(no_reuse_results)

        # ── Configuration B: random scheduling ──
        random_results: list[ExperimentResult] = []
        for _ in range(args.rounds):
            random_results.append(_run_experiment_round(path, args.mock, model, ollama, disable_reuse=False, random_sched=True))
        comp.random_sched = _avg_results(random_results)

        comparisons.append(comp)

    total_elapsed = time.time() - total_start

    if args.json_output:
        output = {
            "total_elapsed_s": round(total_elapsed, 3),
            "scenarios": [
                {
                    "scenario": c.scenario,
                    "rounds": c.rounds,
                    "baseline": None if c.baseline is None else _exp_to_dict(c.baseline),
                    "no_context_reuse": None if c.no_reuse is None else _exp_to_dict(c.no_reuse),
                    "random_scheduling": None if c.random_sched is None else _exp_to_dict(c.random_sched),
                    "improvements": c.improvements(),
                }
                for c in comparisons
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    _print_experiment_report(comparisons, total_elapsed)


def _exp_to_dict(r: ExperimentResult) -> dict[str, Any]:
    return {
        "elapsed_s": round(r.elapsed_s, 4),
        "completed": r.completed,
        "failed": r.failed,
        "segments": r.segments,
        "snapshots": r.snapshots,
        "reuse_ratio": round(r.reuse_ratio, 4),
        "sharing_ratio": round(r.sharing_ratio, 4),
        "total_tokens_est": r.total_tokens_est,
    }


def _print_experiment_report(comparisons: list[ExperimentComparison], total_elapsed: float) -> None:
    sep = "=" * 92
    print(f"\n{sep}")
    print("  Agent Runtime — A/B Experiment Report")
    print(f"  Compares: BASELINE vs NO-REUSE vs RANDOM-SCHEDULING")
    print(f"{sep}")

    for comp in comparisons:
        b = comp.baseline
        nr = comp.no_reuse
        rs = comp.random_sched
        imp = comp.improvements()

        print(f"\n  ── {comp.scenario} ({comp.rounds} rounds each) ──")
        print(f"  {'':<22s} {'time(s)':>8s} {'cmpl':>5s} {'fail':>5s} {'segs':>6s} {'reuse%':>7s} {'tokens':>8s}")
        print(f"  {'-'*22} {'-'*8} {'-'*5} {'-'*5} {'-'*6} {'-'*7} {'-'*8}")

        for label, r in [("BASELINE (reuse+aware)", b), ("NO-REUSE (disable dedup)", nr), ("RANDOM-SCHED (shuffle)", rs)]:
            if r is None:
                continue
            print(
                f"  {label:<22s} {r.elapsed_s:>8.3f} {r.completed:>5d} {r.failed:>5d} "
                f"{r.segments:>6d} {r.reuse_ratio*100:>6.1f}% {r.total_tokens_est:>8d}"
            )

        if imp:
            print(f"\n  Improvements (baseline vs alternatives):")
            if "reuse_segment_savings_pct" in imp:
                print(f"    context reuse → {imp['reuse_segment_savings_pct']:.1f}% fewer segments, "
                      f"{imp['reuse_time_savings_pct']:.1f}% time saved, "
                      f"{imp['reuse_token_savings_pct']:.1f}% token savings")
            if "sched_completion_pct" in imp:
                sched_pct = imp.get("sched_completion_pct", 0)
                time_pct = imp.get("sched_time_savings_pct", 0)
                direction = "more" if sched_pct > 0 else "fewer"
                print(f"    context-aware scheduling → {abs(sched_pct):.1f}% {direction} completions, "
                      f"{abs(time_pct):.1f}% time difference")

    print(f"\n{sep}")
    print(f"  Total experiment elapsed: {total_elapsed:.3f}s")
    print(f"  Run {sum(c.rounds * 3 for c in comparisons)} total benchmark runs")
    print(f"{sep}\n")


def _serve(port: int) -> None:
    import http.server
    import socketserver

    dashboard = Path("dashboard")
    dashboard.mkdir(exist_ok=True)
    if not (dashboard / "index.html").exists():
        source = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
        if source.exists():
            shutil.copy(source, dashboard / "index.html")
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Dashboard serving at http://localhost:{port}/dashboard/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
