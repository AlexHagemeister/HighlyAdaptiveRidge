from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LassoCV


class HAL:
    """Highly Adaptive Lasso (HAL) baseline.

    Explicit-basis implementation using tensor product zero-order splines
    with L1-penalized regression (LassoCV).

    Parameters
    ----------
    **kwargs
        Passed to sklearn.linear_model.LassoCV.
    """

    def __init__(self, **kwargs) -> None:
        self.lasso = LassoCV(**kwargs)
        self.knots: NDArray[np.floating] | None = None
        self.name = "HAL"

    def _basis_products(
        self,
        arr: NDArray[np.bool_],
        index: int = 0,
        current: NDArray[np.bool_] | None = None,
        result: list[NDArray[np.bool_]] | None = None,
    ) -> list[NDArray[np.bool_]]:
        """Recursively enumerate all tensor products of one-way indicator bases."""
        if result is None:
            result = []
        if current is None:
            current = np.ones_like(arr[0], dtype=bool)

        if index == len(arr):
            result.append(current)
        else:
            self._basis_products(arr, index + 1, current & arr[index], result)
            self._basis_products(arr, index + 1, current, result)

        return result

    def _bases(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute the full basis matrix H(X) using stored knots."""
        one_way_bases = np.stack([
            np.less.outer(self.knots[:, j], X[:, j])
            for j in range(self.knots.shape[1])
        ])
        bases = self._basis_products(one_way_bases)
        return np.concatenate(bases[:-1]).T

    def fit(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> None:
        """Fit HAL to training data.

        Parameters
        ----------
        X : ndarray of shape (n, p)
        Y : ndarray of shape (n,)
        """
        self.knots = X
        self.lasso.fit(self._bases(X), Y)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict targets for new data.

        Parameters
        ----------
        X : ndarray of shape (m, p)

        Returns
        -------
        ndarray of shape (m,)
        """
        if self.knots is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.lasso.predict(self._bases(X))
