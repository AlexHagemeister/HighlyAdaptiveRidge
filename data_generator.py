import numpy as np

class DataGenerator:
    """
    This class represents a data generating process (DGP) for generating synthetic data.

    Methods:
    - f(X): Computes the target function value for a given input X.
    - gen(n): Generates synthetic data of size n using the DGP.

    """

    @classmethod
    def f(cls, X):
        """
        Computes the target function value for a given input X.

        Parameters:
        - X: Input data of shape (n, 2), where n is the number of samples.

        Returns:
        - The computed target function values.

        Author: Alejandro Schuler
        Edits: renamed class, added documentation

        """
        return -0.5*X[:,0] + (X[:,1] * X[:,0]**2) / 2.75 + X[:,1]

    @classmethod
    def generate_data(cls, n):
        """
        Generates synthetic data of size n using the data generating process (DGP).

        Parameters:
        - n: The number of samples to generate.

        Returns:
        - X: The generated input data of shape (n, 2).
        - Y: The corresponding target function values of shape (n,).

        """
        X = np.column_stack((
            np.random.uniform(-4, 4, n),
            np.random.binomial(1, 0.5, n),
        ))
        Y = cls.f(X) + np.random.normal(0, 1, n)
        return X, Y