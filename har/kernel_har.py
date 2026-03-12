from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.linalg import solve


class KernelHAR:
    """Kernelized Highly Adaptive Ridge regression.

    Performs ridge regression in a saturated zero-order spline basis using the
    kernel trick from Section 3.2 of the paper:

        K(x, x') = sum_i 2^|s_i(x, x')|

    where s_i(x, x') = {j : X_{i,j} <= min(x_j, x'_j)}.

    Parameters
    ----------
    lambdas : list of float
        Grid of regularization parameters for cross-validation.
    num_folds : int
        Number of CV folds (default 5).
    """

    def __init__(self, lambdas: list[float] | None = None, num_folds: int = 5) -> None:
        self.lambdas = lambdas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.num_folds = num_folds
        self.knots: NDArray[np.floating] | None = None
        self.kernel_matrix: NDArray[np.floating] | None = None
        self.alpha: NDArray[np.floating] | None = None
        self.best_lambda: float | None = None
        self.cv_mses: NDArray[np.floating] | None = None
        self.name = "KernelHAR"

    def _compute_kernel_matrix(
        self, X: NDArray[np.floating], X_prime: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute K(X, X_prime) using the HAR kernel.

        K[i, j] = sum_k 2^|{d : knots[k,d] <= min(X[i,d], X_prime[j,d])}|

        Fully vectorized implementation.
        """
        min_matrix = np.minimum(X[:, np.newaxis, :], X_prime[np.newaxis, :, :])
        comparison = (self.knots[:, np.newaxis, np.newaxis, :] <= min_matrix).sum(axis=-1)
        return np.sum(2 ** comparison, axis=0)

    def fit(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> None:
        """Fit the model with CV-selected regularization.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Training features.
        Y : ndarray of shape (n,)
            Training targets.
        """
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        mse_lists = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            Y_train, Y_val = Y[train_index], Y[val_index]

            self.knots = X_train
            K_train = self._compute_kernel_matrix(X_train, X_train)
            K_val = self._compute_kernel_matrix(X_val, X_train)

            fold_mses = []
            for lam in self.lambdas:
                K_reg = K_train + lam * np.eye(K_train.shape[0])
                alpha = solve(K_reg, Y_train)
                Y_pred = K_val @ alpha
                fold_mses.append(mean_squared_error(Y_val, Y_pred))

            mse_lists.append(fold_mses)

        self.cv_mses = np.mean(mse_lists, axis=0)
        self.best_lambda = self.lambdas[np.argmin(self.cv_mses)]

        # Refit on full data with best lambda
        self.knots = X
        self.kernel_matrix = self._compute_kernel_matrix(X, X)
        K_reg = self.kernel_matrix + self.best_lambda * np.eye(self.kernel_matrix.shape[0])
        self.alpha = solve(K_reg, Y)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict targets for new data.

        Parameters
        ----------
        X : ndarray of shape (m, p)
            Test features.

        Returns
        -------
        ndarray of shape (m,)
            Predicted values.
        """
        if self.alpha is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        K = self._compute_kernel_matrix(X, self.knots)
        return K @ self.alpha
