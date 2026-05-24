from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

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

    args = parser.parse_args()
    if args.cmd == "submit":
        _submit(args)
    elif args.cmd == "inspect":
        _inspect(args.path)
    elif args.cmd == "benchmark":
        _benchmark(args)
    elif args.cmd == "serve":
        _serve(args.port)


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
