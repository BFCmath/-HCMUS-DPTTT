from __future__ import annotations

import csv
import random
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpt_xgboost_baseline import (  # noqa: E402
    ColumnBlock,
    approximate_split,
    column_block_split,
    exact_greedy_split,
    propose_candidates,
    sparsity_aware_split,
)


@dataclass(frozen=True)
class BenchmarkResult:
    samples: int
    features: int
    algorithm: str
    median_seconds: float
    peak_kib: float
    candidate_count: int | None
    split_gain: float | None


def make_dense(samples: int, features: int, seed: int) -> tuple[list[list[float]], list[float], list[float]]:
    rng = random.Random(seed)
    x = [[rng.random() for _ in range(features)] for _ in range(samples)]
    gradients = [rng.gauss(0.0, 1.0) for _ in range(samples)]
    hessians = [0.1 + rng.random() for _ in range(samples)]
    return x, gradients, hessians


def make_sparse(
    samples: int,
    features: int,
    seed: int,
    missing_rate: float = 0.80,
) -> tuple[list[list[float | None]], list[float], list[float]]:
    rng = random.Random(seed)
    x: list[list[float | None]] = []
    for _ in range(samples):
        row: list[float | None] = []
        for _ in range(features):
            row.append(None if rng.random() < missing_rate else rng.random())
        x.append(row)
    gradients = [rng.gauss(0.0, 1.0) for _ in range(samples)]
    hessians = [0.1 + rng.random() for _ in range(samples)]
    return x, gradients, hessians


def measure(call: Callable[[], object], repeats: int) -> tuple[float, float, object]:
    durations: list[float] = []
    peaks: list[float] = []
    last_result: object = None
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        last_result = call()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        durations.append(elapsed)
        peaks.append(peak / 1024.0)
    return statistics.median(durations), statistics.median(peaks), last_result


def count_approx_candidates(
    features: list[list[float]],
    hessians: list[float],
    eps: float,
) -> int:
    indices = list(range(len(features)))
    total = 0
    for feature_index in range(len(features[0])):
        values = [row[feature_index] for row in features]
        candidates = propose_candidates(values, hessians, indices, eps)
        total += max(0, len(candidates) - 1)
    return total


def gain_or_none(result: object) -> float | None:
    return getattr(result, "gain", None)


def run_suite(
    sizes: tuple[int, ...] = (500, 1000, 2000, 5000, 10000),
    features: int = 8,
    repeats: int = 5,
    eps: float = 0.05,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for samples in sizes:
        dense_x, gradients, hessians = make_dense(samples, features, seed=20260615 + samples)
        sparse_x, sparse_gradients, sparse_hessians = make_sparse(
            samples,
            features,
            seed=20261615 + samples,
            missing_rate=0.80,
        )

        dense_candidate_count = features * (samples - 1)
        approx_candidate_count = count_approx_candidates(dense_x, hessians, eps)
        sparse_candidate_count = sum(
            max(0, len({row[j] for row in sparse_x if row[j] is not None}) - 1)
            for j in range(features)
        )

        block = ColumnBlock(dense_x)
        cases: list[tuple[str, int | None, Callable[[], object]]] = [
            (
                "Exact Greedy",
                dense_candidate_count,
                lambda dense_x=dense_x, gradients=gradients, hessians=hessians: exact_greedy_split(
                    dense_x,
                    gradients,
                    hessians,
                ),
            ),
            (
                "Column Block",
                dense_candidate_count,
                lambda block=block, gradients=gradients, hessians=hessians: column_block_split(
                    block,
                    gradients,
                    hessians,
                ),
            ),
            (
                "Approximate",
                approx_candidate_count,
                lambda dense_x=dense_x, gradients=gradients, hessians=hessians: approximate_split(
                    dense_x,
                    gradients,
                    hessians,
                    eps=eps,
                ),
            ),
            (
                "Sparsity-aware",
                sparse_candidate_count,
                lambda sparse_x=sparse_x, sparse_gradients=sparse_gradients, sparse_hessians=sparse_hessians: sparsity_aware_split(
                    sparse_x,
                    sparse_gradients,
                    sparse_hessians,
                ),
            ),
        ]

        for algorithm, candidate_count, call in cases:
            seconds, peak_kib, result = measure(call, repeats)
            results.append(
                BenchmarkResult(
                    samples=samples,
                    features=features,
                    algorithm=algorithm,
                    median_seconds=seconds,
                    peak_kib=peak_kib,
                    candidate_count=candidate_count,
                    split_gain=gain_or_none(result),
                )
            )
    return results


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "samples",
                "features",
                "algorithm",
                "median_seconds",
                "peak_kib",
                "candidate_count",
                "split_gain",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def write_latex_table(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = ["Exact Greedy", "Column Block", "Approximate", "Sparsity-aware"]
    by_size = {
        samples: {row.algorithm: row for row in results if row.samples == samples}
        for samples in sorted({row.samples for row in results})
    }
    lines = [
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "\\textbf{Samples} & \\textbf{Exact} & \\textbf{Column Block} & \\textbf{Approximate} & \\textbf{Sparsity-aware} \\\\",
        "\\midrule",
    ]
    for samples, rows in by_size.items():
        values = [f"{rows[algorithm].median_seconds:.4f}" for algorithm in algorithms]
        lines.append(f"{samples:,} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_runtime_plot(results: list[BenchmarkResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = ["Exact Greedy", "Column Block", "Approximate", "Sparsity-aware"]
    colors = {
        "Exact Greedy": "#1f77b4",
        "Column Block": "#d62728",
        "Approximate": "#2ca02c",
        "Sparsity-aware": "#9467bd",
    }
    samples = sorted({row.samples for row in results})
    by_algorithm = {
        algorithm: [row for row in results if row.algorithm == algorithm]
        for algorithm in algorithms
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for algorithm in algorithms:
        rows = sorted(by_algorithm[algorithm], key=lambda row: row.samples)
        ax.plot(
            samples,
            [row.median_seconds for row in rows],
            marker="o",
            linewidth=2,
            label=algorithm,
            color=colors[algorithm],
        )

    ax.set_title("Split-finding runtime on deterministic synthetic data")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Median runtime (seconds)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    results = run_suite()
    output_dir = ROOT / "benchmarks" / "results"
    write_csv(results, output_dir / "split_finding_benchmark.csv")
    write_latex_table(results, output_dir / "split_finding_runtime_table.tex")
    write_runtime_plot(results, ROOT / "XGBoost" / "images" / "runtime_comparison.pdf")
    for row in results:
        print(
            f"{row.samples:5d} {row.algorithm:15s} "
            f"{row.median_seconds:8.4f}s peak={row.peak_kib:8.1f} KiB "
            f"candidates={row.candidate_count}"
        )


if __name__ == "__main__":
    main()
