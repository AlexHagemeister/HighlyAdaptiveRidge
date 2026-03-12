# Highly Adaptive Ridge (HAR)

Companion code for the paper:

> **Highly Adaptive Ridge**
> Alejandro Schuler, Alexander Hagemeister, Mark van der Laan
> Division of Biostatistics & EECS, UC Berkeley (2024)
> [arXiv:2410.02680](https://arxiv.org/abs/2410.02680)

## What is HAR?

HAR is a nonparametric regression algorithm that performs kernel ridge regression with a data-adaptive kernel derived from a saturated zero-order spline basis expansion. The kernel computes inner products in a high-dimensional basis space without ever instantiating the basis matrix, making the method scalable to high-dimensional covariates.

**Key theoretical result:** HAR achieves an *n*<sup>−1/3</sup>(log *n*)<sup>2(*p*−1)/3</sup> L2 convergence rate in the class of càdlàg functions of bounded sectional variation — a dimension-free rate (up to log factors) in a rich nonparametric function class.

## Quick Start

```bash
git clone https://github.com/AlexHagemeister/HighlyAdaptiveRidge.git
cd HighlyAdaptiveRidge
pip install -e .

# Reproduce simulation experiments (convergence rate + timing)
python experiments/run_simulations.py

# Reproduce UCI benchmark results
python experiments/run_benchmarks.py

# Generate figures
python experiments/plot_results.py
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Repository Structure

```
har/                      # Core implementations
├── kernel_har.py         # Kernelized HAR (paper's main algorithm)
├── hal.py                # Highly Adaptive Lasso (baseline)
└── data_generators.py    # Synthetic DGPs (Smooth, Jump, Sinusoidal)

experiments/              # Reproducibility scripts
├── run_simulations.py    # Convergence + timing on synthetic data
├── run_benchmarks.py     # UCI real-data evaluation
└── plot_results.py       # Figure generation

tests/                    # Test suite
data/                     # UCI benchmark datasets
results/figures/          # Generated plots
docs/                     # Paper and supplementary material
```

## How the Kernel Works

The HAR kernel between points *x* and *x'* given training data {*X*<sub>1</sub>, ..., *X*<sub>*n*</sub>} is:

*K*(*x*, *x'*) = Σ<sub>*i*</sub> 2<sup>|*s*<sub>*i*</sub>(*x*, *x'*)|</sup>

where *s*<sub>*i*</sub>(*x*, *x'*) = {*j* : *X*<sub>*i*,*j*</sub> ≤ min(*x*<sub>*j*</sub>, *x'*<sub>*j*</sub>)}. This avoids instantiating the *n*·2<sup>*p*</sup> basis functions and reduces training to an *O*(*n*<sup>3</sup>) kernel ridge regression.

## Paper

The full paper is available at [docs/HAR_Paper.md](docs/HAR_Paper.md).

## Citation

```bibtex
@article{schuler2024har,
  title={Highly Adaptive Ridge},
  author={Schuler, Alejandro and Hagemeister, Alexander and van der Laan, Mark},
  year={2024}
}
```
