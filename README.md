# XGBoost Split-Finding Algorithms Baseline

This repository contains a Python implementation of the algorithms for split finding described in the original XGBoost paper. It was created as part of the HCMUS DPTTT program.

## Features

The implementation covers several tree-building algorithms:
- **Exact Greedy Algorithm** (`exact_greedy.py`): Implements the dense exact split finder (Algorithm 1) from the XGBoost paper.
- **Sparsity-aware Split Finding** (`sparsity_aware.py`): Handles missing values and sparse data structures (Algorithm 3).
- **Approximate Algorithm** (`approximate.py`): Histogram-based approximate split finding using quantiles (Algorithm 2).
- **Column Block Structure** (`column_block.py`): Support for cache-aware and parallelized node splitting.

## Installation

You can set up a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

## API Usage

The core implementation lives in `src/dpt_xgboost_baseline/`. The primary API across modules typically includes functions like:

- `*_split(features, gradients, hessians, instance_indices=None, config=None)`: Computes the optimal split according to the respective algorithm.
- `split_gain(...)`: Calculates the gain for a potential split.
- `leaf_weight(...)`: Computes the optimal leaf weight.

> **Note**: For `exact_greedy_split`, the `features` argument must be a dense row-major matrix. Missing or non-finite values are rejected by the basic exact greedy baseline; use the sparsity-aware module for such data.

## Testing

Run the test suite using `pytest`:

```bash
pytest -v
```

## Benchmarks

The benchmark script compares the performance of the implemented split-finding algorithms and generates a report.

To run the benchmark:

```bash
python benchmarks/split_finding_benchmark.py
```

The results are exported as CSV files and LaTeX tables under the `benchmarks/results/` directory.
