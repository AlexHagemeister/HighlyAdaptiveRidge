import numpy as np

class DataGenerator:
    """
    This class represents a data generating process (DGP) for generating synthetic data for HAL.

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


# ------------------------------------------------------------------------- # 
#               "Smooth" data generating process (DGP)                      #
# ------------------------------------------------------------------------- # 

class SmoothDataGenerator:
    """
    This class represents a data generating process (DGP) for generating synthetic data for HAL,
    using smooth regression functions as described in the paper.

    Methods:
    - f_1, f_3, f_5: Compute the smooth target function values for dimensions 1, 3, and 5 respectively.
    - generate_data: Generates synthetic data using the DGP based on the specified dimension.
    """

    @staticmethod
    def f_1(X):
        """
        Computes the smooth target function value for dimension 1.

        Parameters:
        - X: Input data of shape (n, 1), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 1.
        """
        x1 = X[:, 0]
        return 0.05 * x1 + 0.04 * x1**2
    
    @staticmethod
    def f_3(X):
        """
        Computes the smooth target function value for dimension 3.

        Parameters:
        - X: Input data of shape (n, 3), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 3.
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return 0.07 * x1 - 0.28 * x1**2 + 0.05 * x2 + 0.25 * x2 * x3
    
    @staticmethod
    def f_5(X):
        """
        Computes the smooth target function value for dimension 5.

        Parameters:
        - X: Input data of shape (n, 5), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 5.
        """
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        return (0.1 * x1 - 0.3 * x1 ** 2 + 0.25 * x2 +
                0.5 * x2 * x3 - 0.5 * x4 + 0.04 * x5 ** 2 - 
                0.1 * x5)

    @staticmethod
    def generate_data(n, d):
        """
        Generates synthetic data using the smooth data generating process (DGP) for a specified dimension.

        Parameters:
        - n: The number of samples to generate.
        - d: The dimension of the data to generate (1, 3, or 5).

        Returns:
        - X: The generated input data of shape (n, d).
        - Y: The corresponding target function values of shape (n,).
        """

        # Define the distributions for X based on the paper
        distributions = {
            1: lambda size: np.random.uniform(-4, 4, size),
            2: lambda size: np.random.uniform(-4, 4, size),
            3: lambda size: np.random.binomial(1, 0.5, size),
            4: lambda size: np.random.normal(0, 1, size),
            5: lambda size: np.random.gamma(2, 1, size),
        }

        # Generate features X for the specified dimension d
        X = np.column_stack([distributions[j + 1](n) for j in range(d)])
        
        # Apply the appropriate target function based on dimension d
        if d == 1:
            Y = SmoothDataGenerator.f_1(X)
        elif d == 3:
            Y = SmoothDataGenerator.f_3(X)
        elif d == 5:
            Y = SmoothDataGenerator.f_5(X)
        else:
            raise ValueError("Dimension d must be either 1, 3, or 5.")

        # Add normal noise to the target function values
        Y += np.random.normal(0, 1, n)

        return X, Y

# ------------------------------------------------------------------------- # 
#                                "Jump" DGP                                 #
# ------------------------------------------------------------------------- # 

class JumpDataGenerator:
    """
    This class represents a data generating process (DGP) for generating synthetic data using
    jump regression functions as defined in the paper.

    Methods:
    - f_1, f_3, f_5: Compute the jump target function values for dimensions 1, 3, and 5 respectively.
    - generate_data: Generates synthetic data using the DGP based on the specified dimension.
    """

    @staticmethod
    def f_1(X):
        """
        Computes the jump target function value for dimension 1.

        Parameters:
        - X: Input data of shape (n, 1), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 1.
        """
        x1 = X[:, 0]
        return (-2.7 * (x1 < -3) + 2.5 * ((x1 > -2) & (x1 <= 0)) - 
                2 * (x1 > 0) + 4 * (x1 > 2) - 3 * (x1 > 3))

    @staticmethod
    def f_3(X):
        """
        Computes the jump target function value for dimension 3.

        Parameters:
        - X: Input data of shape (n, 3), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 3.
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

        return (-2*(x1 < -3)*x3 + 2.5*(x1 > -2) - 2*(x1 > 0) + 
                2.5*(x1 > 2)*x3 - 2.5 * (x1 > 3) + (x2 > -1) - 
                4*(x2 > 1)*x3 + 2*(x2 > 3))

    @staticmethod
    def f_5(X):
        """
        Computes the jump target function value for dimension 5.

        Parameters:
        - X: Input data of shape (n, 5), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 5.
        """
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        return(- (x1 < -3)*x3 + 0.5*(x1 > -2) - (x1 > 0) +
               2*(x1 > 2)*x3 - 3*(x1 > 3) + 1.5*(x2 > -1) -
                5*(x2 > 1)*x3 + 2*(x2 > 3) + 2*(x4 < 0) -
                (x5 > 5) - (x4 < 0)*(x1 < 0) + 2*x3)

    @staticmethod
    def generate_data(n, d):
        """
        Generates synthetic data using the jump data generating process (DGP) for a specified dimension.

        Parameters:
        - n: The number of samples to generate.
        - d: The dimension of the data to generate (1, 3, or 5).

        Returns:
        - X: The generated input data of shape (n, d).
        - Y: The corresponding target function values of shape (n,).
        """
        # Define the distributions for X based on the paper
        distributions = {
            1: lambda size: np.random.uniform(-4, 4, size),
            2: lambda size: np.random.uniform(-4, 4, size),
            3: lambda size: np.random.binomial(1, 0.5, size),
            4: lambda size: np.random.normal(0, 1, size),
            5: lambda size: np.random.gamma(2, 1, size),
        }

        # Generate features X for the specified dimension d
        X = np.column_stack([distributions[j + 1](n) for j in range(d)])
        
        # Apply the appropriate target function based on dimension d
        if d == 1:
            Y = SmoothDataGenerator.f_1(X)
        elif d == 3:
            Y = SmoothDataGenerator.f_3(X)
        elif d == 5:
            Y = SmoothDataGenerator.f_5(X)
        else:
            raise ValueError("Dimension d must be either 1, 3, or 5.")

        # Add normal noise to the target function values
        Y += np.random.normal(0, 1, n)

        return X, Y


# ------------------------------------------------------------------------- # 
#                                "Sinusoidal" DGP                           #
# ------------------------------------------------------------------------- # 
    
import numpy as np

class SinusoidalDataGenerator:
    """
    This class represents a data generating process (DGP) for generating synthetic data using
    sinusoidal regression functions as defined in the paper.

    Methods:
    - f_1, f_3, f_5: Compute the sinusoidal target function values for dimensions 1, 3, and 5 respectively.
    - generate_data: Generates synthetic data using the DGP based on the specified dimension.
    """

    @staticmethod
    def f_1(X):
        """
        Computes the sinusoidal target function value for dimension 1.

        Parameters:
        - X: Input data of shape (n, 1), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 1.
        """
        x1 = X[:, 0]
        return 2 * np.sin(0.5 * np.pi * np.abs(x1)) + 2 * np.cos(0.5 * np.pi * np.abs(x1))

    @staticmethod
    def f_3(X):
        """
        Computes the sinusoidal target function value for dimension 3.

        Parameters:
        - X: Input data of shape (n, 3), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 3.
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        # The indicator function is implemented using numpy's where method
        return (4 * x3 * np.where(x2 < 0, np.sin(0.5 * np.pi * np.abs(x1)), 0) +
                4.1 * np.where(x2 >= 0, np.cos(0.5 * np.pi * np.abs(x1)), 0))

    @staticmethod
    def f_5(X):
        """
        Computes the sinusoidal target function value for dimension 5.

        Parameters:
        - X: Input data of shape (n, 5), where n is the number of samples.

        Returns:
        - The computed target function values for dimension 5.
        """
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        # The indicator function is implemented using numpy's where method
        return (3.8 * x3 * np.where(x2 < 0, np.sin(0.5 * np.pi * np.abs(x1)), 0) +
                4.1 * np.where(x2 > 0, np.cos(np.pi * np.abs(x1) / 2), 0) +
                0.1 * x5 * np.sin(np.pi * x4) + 
                x3 * np.cos(np.abs(x4 - x5)))

    @staticmethod
    def generate_data(n, d):
        """
        Generates synthetic data using the sinusoidal data generating process (DGP) for a specified dimension.

        Parameters:
        - n: The number of samples to generate.
        - d: The dimension of the data to generate (1, 3, or 5).

        Returns:
        - X: The generated input data of shape (n, d).
        - Y: The corresponding target function values of shape (n,).
        """
        # Define the distributions for X based on the paper
        distributions = {
            1: lambda size: np.random.uniform(-4, 4, size),
            2: lambda size: np.random.uniform(-4, 4, size),
            3: lambda size: np.random.binomial(1, 0.5, size),
            4: lambda size: np.random.normal(0, 1, size),
            5: lambda size: np.random.gamma(2, 1, size),
        }

        # Generate features X for the specified dimension d
        X = np.column_stack([distributions[j + 1](n) for j in range(d)])
        
        # Apply the appropriate target function based on dimension d
        if d == 1:
            Y = SmoothDataGenerator.f_1(X)
        elif d == 3:
            Y = SmoothDataGenerator.f_3(X)
        elif d == 5:
            Y = SmoothDataGenerator.f_5(X)
        else:
            raise ValueError("Dimension d must be either 1, 3, or 5.")

        # Add normal noise to the target function values
        Y += np.random.normal(0, 1, n)

        return X, Y
