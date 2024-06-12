import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.linalg import inv
from scipy.linalg import solve

class KernelHAR:
    def __init__(self, *args, **kwargs):
        """
        Initialize the KernelHAR object with optional keyword arguments.

        Keyword Args:
            lambdas (list): Optional grid of lambda values for ridge regression.
                            Defaults to [0.1, 1.0, 10.0].
        """
        self.knots = None  # C - The knots are set to None initially
        self.lambdas = kwargs.get('lambdas', [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])  # λ - Regularization parameters
        self.num_folds = kwargs.get('num_folds', 5)  # Number of folds for cross-validation
        self.kernel_matrix = None  # K - Kernel matrix placeholder
        self.alpha = None  # α - Coefficients placeholder
        self.best_lambda = None  # Best λ - Placeholder for the best lambda found through cross-validation
        self.name = "KernelHAR"
        self.cv_mses = None


    def _compute_kernel_matrix(self, X, X_prime):
        """
        Computes the kernel matrix for the input data X and X̃.

        Args:
            X (ndarray): Input data.
            X̃ (ndarray): Data for which the kernel matrix is computed.

        Returns:
            ndarray: Kernel matrix.
        """
        # OLD - Slow implementation
        # return np.sum([2 ** np.sum(x_i <= np.minimum(x, x_prime)) for x_i in self.knots])

        # Hopefully faster implementation
        # Compute the minimum matrix once
        min_matrix = np.minimum(X[:, np.newaxis, :], X_prime[np.newaxis, :, :])
        print("built min matrix")
        # Compute the comparison result
        comparison = (self.knots[:, np.newaxis, np.newaxis, :] <= min_matrix).sum(axis=-1)
        print("built comparison matrix")
        # Compute the kernel values using vectorized operations
        return np.sum(2 ** comparison, axis=0)

    def fit(self, X, Y):
        """
        Fit the HAR model using the provided training data with cross-validation to select the best lambda.

        This involves:
        1. Setting the knots to the training data.
        2. Computing the kernel matrix for the training data.
        3. Performing cross-validation to find the best regularization parameter (lambda).
        4. Solving the ridge regression problem with the best lambda to obtain the coefficients.

        Args:
            X (ndarray): Training data.
            Y (ndarray): Target values.
        """

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)  # using sklearn KFold for cross-validation
        mse_lists = [] # list of MSEs for each lambda value for each fold

        # Iterate over each fold and each lambda value.
        # NOTE: Kf.split(X) returns the indices of the training and validation sets for each fold.
        for train_index, val_index in kf.split(X):

            # Split the data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            Y_train, Y_val = Y[train_index], Y[val_index]

            self.knots = X_train  # C 
            print("computing kernel matrix for K_train for fold: ", train_index)
            K_train = self._compute_kernel_matrix(X_train, X_train)
            print("K_train kernel matrix computed for fold: ", train_index)
            print("computing kernel matrix for K_val for fold: ", val_index)
            K_val = self._compute_kernel_matrix(X_val, X_train)
            print("K_val kernel matrix computed for fold: ", val_index)

            fold_mses = [] # List to store the MSE for each lambda value for the current fold

            # For each lambda value, compute the MSE for the validation set
            for λ in self.lambdas:

                K_reg = K_train + λ * np.eye(K_train.shape[0]) # compute the regularized kernel matrix and solve 
                print("computed regularized kernel matrix for fold: ", train_index, " and lambda: ", λ)
                α = solve(K_reg, Y_train)
                print("solved for alpha for fold: ", train_index, " and lambda: ", λ)
                
                Y_pred = K_val @ α # Predict the target values for the validation set
                print("predicted Y for fold: ", val_index, " and lambda: ", λ)
                fold_mses.append(mean_squared_error(Y_val, Y_pred)) # Save the MSE for the validation set
            
            mse_lists.append(fold_mses) # Store the list of MSEs for each lambda value for each fold

        self.cv_mses = np.mean(mse_lists, axis=0)  # Compute the average MSE for each lambda value across all folds

        self.best_lambda = self.lambdas[np.argmin(self.cv_mses)] # Find the best lambda value that minimizes the average MSE
        print(f"Best lambda: {self.best_lambda}")
        
        # Fit the model on the entire dataset with the best lambda
        self.knots = X  # update the knots to the entire dataset
        self.kernel_matrix = self._compute_kernel_matrix(X, X)
        print("computed kernel matrix for the entire dataset")

        K_reg = self.kernel_matrix + self.best_lambda * np.eye(self.kernel_matrix.shape[0])
        self.alpha = solve(K_reg, Y)
        print("solved for alpha for the entire dataset")

    def predict(self, X̃):
        """
        Predict the target values for the given data using the best lambda.

        Args:
            X̃ (ndarray): Data for which predictions are made.

        Returns:
            ndarray: Predicted values.
        """
        if self.alpha is None:
            raise ValueError("The model is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        K̃ = self._compute_kernel_matrix(X̃, self.knots)  # K
        return K̃ @ self.alpha  # α

