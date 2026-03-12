from sklearn.linear_model import RidgeCV
import numpy as np

from kernel_har import KernelHAR

class HAR:
    """
    Highly Adaptive Ridge (HAR) class.

    This class implements the Highly Adaptive Ridge algorithm, which is a variant of the Ridge algorithm.
    HAR is used for feature selection and regression tasks with an emphasis on controlling the L2 norm of coefficients.

    Attributes:
        ridge (RidgeCV): RidgeCV object for performing Ridge regression.
        knots (ndarray): Array of knot points used for basis functions.

    Methods:
        __init__(self, *args, **kwargs): Initializes the HAR object.
        _basis_products(self, arr, index=0, current=None, result=None): Recursive helper function for computing basis products.
        _bases(self, X): Computes the basis functions for the given input data.
        fit(self, X, Y): Fits the HAR model to the given input data and target values.
        predict(self, X): Predicts the target values for the given input data.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the HAR object.
        """
        self.ridge = RidgeCV(*args, **kwargs)
        self.name = "HAR"
        self.best_lambda = None # update with best lambda found through cross-validation

    def _basis_products(self, arr, index=0, current=None, result=None):
        # Implementation remains the same as in HAL
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
        pass

    def _bases(self, X):
        # Implementation remains the same as in HAL
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
        pass

    def fit(self, X, Y):
        """
        Fits the HAR model to the given input data and target values.
        """
        self.knots = X
        self.ridge.fit(self._bases(X), Y) # Adjusted for Ridge
        # update best lambda found through cross-validation
        self.best_lambda = self.ridge.alpha_

    def predict(self, X):
        """
        Predicts the target values for the given input data.
        """
        return self.ridge.predict(self._bases(X)) # Adjusted for Ridge
