"""Data generating processes (DGPs) for HAR simulation experiments.

Three DGP families from the paper: Smooth, Jump, Sinusoidal.
Each has explicit target functions for d=1, 3, 5 and a general formula
for arbitrary d.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Feature distributions shared across all DGPs (from the paper).
_DISTRIBUTIONS = {
    1: lambda rng, n: rng.uniform(-4, 4, n),
    2: lambda rng, n: rng.uniform(-4, 4, n),
    3: lambda rng, n: rng.binomial(1, 0.5, n),
    4: lambda rng, n: rng.standard_normal(n),
    5: lambda rng, n: rng.gamma(2, 1, n),
}


def _generate_X(rng: np.random.Generator, n: int, d: int) -> NDArray[np.floating]:
    """Generate feature matrix X with the paper's marginal distributions."""
    return np.column_stack([_DISTRIBUTIONS[(j % 5) + 1](rng, n) for j in range(d)])


# ------------------------------------------------------------------ #
#  Smooth DGP                                                         #
# ------------------------------------------------------------------ #

class SmoothDataGenerator:
    name = "Smooth"

    @staticmethod
    def f_1(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """f(x) = 0.05*x1 + 0.04*x1^2."""
        x1 = X[:, 0]
        return 0.05 * x1 + 0.04 * x1**2

    @staticmethod
    def f_3(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """f(x) = 0.07*x1 - 0.28*x1^2 + 0.05*x2 + 0.25*x2*x3."""
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return 0.07 * x1 - 0.28 * x1**2 + 0.05 * x2 + 0.25 * x2 * x3

    @staticmethod
    def f_5(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """f(x) = polynomial with linear, quadratic, and interaction terms in x1..x5."""
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        return (0.1 * x1 - 0.3 * x1**2 + 0.25 * x2 +
                0.5 * x2 * x3 - 0.5 * x4 + 0.04 * x5**2 - 0.1 * x5)

    @staticmethod
    def f_general(X: NDArray[np.floating], d: int) -> NDArray[np.floating]:
        """Alternating linear and quadratic terms for arbitrary d."""
        Y = np.zeros(X.shape[0])
        for i in range(d):
            if i % 2 == 0:
                Y += 0.1 * X[:, i] - 0.2 * X[:, i]**2
            else:
                Y += 0.05 * X[:, i]
        return Y

    @staticmethod
    def generate_data(
        n: int, d: int, seed: int | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate n samples from the Smooth DGP in d dimensions."""
        rng = np.random.default_rng(seed)
        X = _generate_X(rng, n, d)
        f = {1: SmoothDataGenerator.f_1,
             3: SmoothDataGenerator.f_3,
             5: SmoothDataGenerator.f_5}.get(d)
        Y = f(X) if f else SmoothDataGenerator.f_general(X, d)
        Y += rng.standard_normal(n)
        return X, Y


# ------------------------------------------------------------------ #
#  Jump DGP                                                            #
# ------------------------------------------------------------------ #

class JumpDataGenerator:
    name = "Jump"

    @staticmethod
    def f_1(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Piecewise constant with 5 jumps in x1."""
        x1 = X[:, 0]
        return (-2.7 * (x1 < -3) + 2.5 * ((x1 > -2) & (x1 <= 0)) -
                2 * (x1 > 0) + 4 * (x1 > 2) - 3 * (x1 > 3))

    @staticmethod
    def f_3(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Conditional step functions with x3 multiplier."""
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return (-2 * (x1 < -3) * x3 + 2.5 * (x1 > -2) - 2 * (x1 > 0) +
                2.5 * (x1 > 2) * x3 - 2.5 * (x1 > 3) + (x2 > -1) -
                4 * (x2 > 1) * x3 + 2 * (x2 > 3))

    @staticmethod
    def f_5(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Step functions with interactions across x1..x5."""
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        return (-1 * (x1 < -3) * x3 + 0.5 * (x1 > -2) - 1 * (x1 > 0) +
                2 * (x1 > 2) * x3 - 3 * (x1 > 3) + 1.5 * (x2 > -1) -
                5 * (x2 > 1) * x3 + 2 * (x2 > 3) + 2 * (x4 < 0) -
                (x5 > 5) - (x4 < 0) * (x1 < 0) + 2 * x3)

    @staticmethod
    def f_general(
        X: NDArray[np.floating], d: int, rng: np.random.Generator | None = None
    ) -> NDArray[np.floating]:
        """Random indicator functions with uniform coefficients."""
        if rng is None:
            rng = np.random.default_rng(42)
        coeffs = rng.uniform(-5, 5, d)
        Y = np.zeros(X.shape[0])
        for i in range(d):
            Y += coeffs[i] * (X[:, i] > 0)
        return Y

    @staticmethod
    def generate_data(
        n: int, d: int, seed: int | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate n samples from the Jump DGP in d dimensions."""
        rng = np.random.default_rng(seed)
        X = _generate_X(rng, n, d)
        f = {1: JumpDataGenerator.f_1,
             3: JumpDataGenerator.f_3,
             5: JumpDataGenerator.f_5}.get(d)
        Y = f(X) if f else JumpDataGenerator.f_general(X, d, rng=rng)
        Y += rng.standard_normal(n)
        return X, Y


# ------------------------------------------------------------------ #
#  Sinusoidal DGP                                                     #
# ------------------------------------------------------------------ #

class SinusoidalDataGenerator:
    name = "Sinusoidal"

    @staticmethod
    def f_1(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """f(x) = 2*sin(pi*|x1|/2) + 2*cos(pi*|x1|/2)."""
        x1 = X[:, 0]
        return 2 * np.sin(0.5 * np.pi * np.abs(x1)) + 2 * np.cos(0.5 * np.pi * np.abs(x1))

    @staticmethod
    def f_3(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Conditional sin/cos based on sign of x2."""
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return (4 * x3 * np.where(x2 < 0, np.sin(0.5 * np.pi * np.abs(x1)), 0) +
                4.1 * np.where(x2 >= 0, np.cos(0.5 * np.pi * np.abs(x1)), 0))

    @staticmethod
    def f_5(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Multiple trigonometric interactions across x1..x5."""
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        return (3.8 * x3 * np.where(x2 < 0, np.sin(0.5 * np.pi * np.abs(x1)), 0) +
                4.1 * np.where(x2 > 0, np.cos(np.pi * np.abs(x1) / 2), 0) +
                0.1 * x5 * np.sin(np.pi * x4) +
                x3 * np.cos(np.abs(x4 - x5)))

    @staticmethod
    def f_general(
        X: NDArray[np.floating], d: int, rng: np.random.Generator | None = None
    ) -> NDArray[np.floating]:
        """Random sinusoidal combinations of feature subsets."""
        if rng is None:
            rng = np.random.default_rng(42)
        p = X.shape[1]
        Y = np.zeros(X.shape[0])
        for _ in range(d):
            subset_size = min(1 + rng.poisson(1), p)
            subset = rng.choice(p, subset_size, replace=False)
            term = np.prod(X[:, subset], axis=1)
            coef_sin = rng.uniform(-2, 2)
            coef_cos = rng.uniform(-2, 2)
            Y += coef_sin * np.sin(np.pi * np.abs(term) / 2) + coef_cos * np.cos(np.pi * np.abs(term) / 2)
        return Y

    @staticmethod
    def generate_data(
        n: int, d: int, seed: int | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate n samples from the Sinusoidal DGP in d dimensions."""
        rng = np.random.default_rng(seed)
        X = _generate_X(rng, n, d)
        f = {1: SinusoidalDataGenerator.f_1,
             3: SinusoidalDataGenerator.f_3,
             5: SinusoidalDataGenerator.f_5}.get(d)
        Y = f(X) if f else SinusoidalDataGenerator.f_general(X, d, rng=rng)
        Y += rng.standard_normal(n)
        return X, Y
