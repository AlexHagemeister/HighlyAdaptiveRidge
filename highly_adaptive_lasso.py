# Highly Adaptive Lasso (simpler). Unsure about the basis expansion. 
# Only currently seems to run on data which has already undergone a basis expansion.

from sklearn.linear_model import Lasso, LassoCV, RidgeCV
import numpy as np

class HAL:
    """
    Highly Adaptive Lasso (HAL) class.

    This class implements the Highly Adaptive Lasso algorithm, which is a variant of the Lasso algorithm.
    HAL is used for feature selection and regression tasks.

    Attributes:
        lasso (LassoCV): LassoCV object for performing Lasso regression.
        knots (ndarray): Array of knot points used for basis functions.

    Methods:
        __init__(self, *args, **kwargs): Initializes the HAL object.
        _basis_products(self, arr, index=0, current=None, result=None): Recursive helper function for computing basis products.
        _bases(self, X): Computes the basis functions for the given input data.
        fit(self, X, Y): Fits the HAL model to the given input data and target values.
        predict(self, X): Predicts the target values for the given input data.

    Author: Alejandro Schuler
    Edits: added documentation

    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the HAL object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        self.lasso = LassoCV(*args, **kwargs)

    def _basis_products(self, arr, index=0, current=None, result=None):
        """
        Recursive helper function for computing basis products.

        This function computes the basis products for the given array of boolean values.

        Args:
            arr (ndarray): Array of boolean values.
            index (int): Current index for recursion (default: 0).
            current (ndarray): Current basis product (default: None).
            result (list): List to store the computed basis products (default: None).

        Returns:
            list: List of computed basis products.

        """
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

    def _bases(self, X):
        """
        Computes the basis functions for the given input data.

        This function computes the basis functions using the knot points and the input data.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Array of computed basis functions.

        """
        one_way_bases = np.stack([
            np.less.outer(self.knots[:,j], X[:,j])
            for j in range(self.knots.shape[1])
        ])
        bases = self._basis_products(one_way_bases)
        return np.concatenate(bases[:-1]).T

    def fit(self, X, Y):
        """
        Fits the HAL model to the given input data and target values.

        This function fits the HAL model to the given input data and target values.

        Args:
            X (ndarray): Input data.
            Y (ndarray): Target values.

        """
        self.knots = X
        self.lasso.fit(self._bases(X), Y) # (HᵀH + λI)^{-1}HᵀY = β

    def predict(self, X):
        """
        Predicts the target values for the given input data.

        This function predicts the target values for the given input data.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Array of predicted target values.

        """
        return self.lasso.predict(self._bases(X)) # Hβ
