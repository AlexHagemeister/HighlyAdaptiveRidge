# Different, more verbose HAL implementation.
# TODO: Ask Alejandro 

from collections import defaultdict
from itertools import chain
from sklearn.linear_model import LassoCV
import numpy as np

from data_generator import DataGenerator

def quantize_col(x, k):
    """
    Quantizes the values in array x into k equally spaced bins.

    Parameters:
    x (ndarray): Input array to be quantized.
    k (int): Number of bins to quantize the values into.

    Returns:
    ndarray: Quantized array with values replaced by their corresponding bin values.
    """
    if k == 0:
        return np.full_like(x, np.min(x))
    quantiles = np.quantile(x, np.arange(0, 1, 1/k))
    indices = np.searchsorted(quantiles, x, side='right') - 1
    return quantiles[indices]

def quantize(X, k):
    """
    Quantizes the input matrix X by reducing the number of unique values in each column to k.

    Parameters:
    X (numpy.ndarray): The input matrix of shape (n_samples, n_features).
    k (int): The desired number of unique values in each column after quantization.

    Returns:
    numpy.ndarray: The quantized matrix of shape (n_samples, n_features).
    """
    if k >= X.shape[0]:
        return X
    return np.stack([quantize_col(x, k) for x in X.T]).T

class HAL:
    """
    HAL (Hierarchical Adaptive Lasso) class for feature selection and prediction.

    Parameters:
    - bin_depths (dict): A dictionary specifying the bin depths for each number of bins. 
                         The keys represent the number of bins, and the values are lists of depths.
                         If None, the default bin depths are {np.inf: []}.
    - sparse_cutoff (float): The cutoff value for sparsity. Default is None.
    - filter (bool): Whether to apply feature filtering. Default is False.
    - **kwargs: Additional keyword arguments to be passed to the LassoCV class.

    Methods:
    - fit_val(X, Y): Fit the HAL model on the training data X and target variable Y.
    - predict(X): Make predictions using the HAL model on the input data X.
    """
    def __init__(self, bin_depths=None, sparse_cutoff=None, filter=False, **kwargs):
        self.lasso = LassoCV(**kwargs)
        self.sparse_cutoff = sparse_cutoff
        self.filter = filter
        self.filter_idx = slice(None)
        if bin_depths is None: # {n_bin: (depths)}
            self.bin_depths = {np.inf: []}
        else:
            self.bin_depths = bin_depths

    @classmethod
    def _basis_products(cls, one_way, max_depth=None, index=0, basis=None, bases=None):
        if max_depth is None:
            max_depth = len(one_way)
        if bases is None:
            bases = defaultdict(list)
        if basis is None:
            basis = np.ones_like(one_way[0], dtype=bool)

        if index == len(one_way) or max_depth == 0:
            bases[max_depth].append(basis)
        else:
            cls._basis_products(one_way, max_depth-1, index+1, basis & one_way[index], bases)
            cls._basis_products(one_way, max_depth,   index+1, basis,                  bases)
        return bases

    @classmethod
    def _one_way(cls, X, knots):
        return np.stack([
            np.less_equal.outer(knots[:,j], X[:,j])
            for j in range(knots.shape[1])
        ])

    @classmethod
    def bases(cls, X, knots, depths):
        if len(depths)==0: # [] represents all depths
            depths = range(1, X.shape[1]+1)
        bases = cls._basis_products(cls._one_way(X, knots), max(depths))
        return np.concatenate(list(chain.from_iterable(
            [bases[max(depths)-d] for d in depths]
        )))

    def multibases(self, X):
        return np.concatenate([
            self.bases(X, knots, depths)
            for (depths, knots) in self.knots.items()
        ]).T

    def filter_bases(self, H):
        if self.sparse_cutoff is None:
            self.sparse_cutoff = 1/np.sqrt(H.shape[0])
        H, filter_idx = np.unique(H, return_index=True, axis=1)
        pct1 = np.mean(H, axis=0)
        keep = (self.sparse_cutoff < pct1) & (pct1 < 1-self.sparse_cutoff)
        self.filter_idx = filter_idx[keep]
        return H[:,keep]

    def fit_val(self, X, Y):
        self.knots = {
            tuple(depths): quantize(X, n_bin)
            for (n_bin, depths) in self.bin_depths.items()
        }
        H = self.multibases(X)
        if self.filter:
            H = self.filter_bases(H)

        self.lasso.fit(H, Y)

    def predict(self, X):
        H = self.multibases(X)[:,self.filter_idx]
        return self.lasso.predict(H)

def HAL9001(**kwargs):
    """
    This function creates an instance of the HAL class with predefined bin depths.
    
    Parameters:
    - kwargs: Additional keyword arguments to be passed to the HAL constructor.
    
    Returns:
    - An instance of the HAL class.
    """
    return HAL(bin_depths = {200/2**(d-1):[d] for d in range(1,4)}, **kwargs)
