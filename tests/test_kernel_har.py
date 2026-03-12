import numpy as np
import pytest

from har.kernel_har import KernelHAR


@pytest.fixture
def simple_data():
    """Small 1-d dataset: y = 2x, n=30."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 1))
    Y = 2 * X[:, 0]
    return X, Y


class TestFitPredict:
    def test_predict_shape(self, simple_data):
        X, Y = simple_data
        model = KernelHAR(num_folds=2)
        model.fit(X, Y)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_fit_sets_attributes(self, simple_data):
        X, Y = simple_data
        model = KernelHAR(num_folds=2)
        model.fit(X, Y)
        assert model.alpha is not None
        assert model.best_lambda is not None
        assert model.knots is not None

    def test_predict_before_fit(self):
        model = KernelHAR()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([[1.0]]))


class TestKernel:
    def test_kernel_symmetry(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (20, 2))
        model = KernelHAR()
        model.knots = X
        K = model._compute_kernel_matrix(X, X)
        np.testing.assert_allclose(K, K.T)

    def test_kernel_positivity(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (20, 2))
        model = KernelHAR()
        model.knots = X
        K = model._compute_kernel_matrix(X, X)
        assert np.all(K > 0)


class TestCVLambda:
    def test_best_lambda_in_grid(self, simple_data):
        X, Y = simple_data
        lambdas = [0.1, 1.0, 10.0]
        model = KernelHAR(lambdas=lambdas, num_folds=2)
        model.fit(X, Y)
        assert model.best_lambda in lambdas

    def test_cv_mses_length(self, simple_data):
        X, Y = simple_data
        lambdas = [0.1, 1.0, 10.0]
        model = KernelHAR(lambdas=lambdas, num_folds=2)
        model.fit(X, Y)
        assert len(model.cv_mses) == len(lambdas)


class TestOverfitting:
    def test_fits_linear_function(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(-1, 1, (50, 1))
        Y = 3 * X[:, 0] + 1
        model = KernelHAR(num_folds=2)
        model.fit(X, Y)
        preds = model.predict(X)
        mse = np.mean((preds - Y) ** 2)
        assert mse < 0.5


class TestDeterminism:
    def test_same_seed_same_result(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(0, 1, (30, 2))
        Y = X[:, 0] + X[:, 1]

        m1 = KernelHAR(num_folds=2)
        m1.fit(X, Y)
        p1 = m1.predict(X[:5])

        m2 = KernelHAR(num_folds=2)
        m2.fit(X, Y)
        p2 = m2.predict(X[:5])

        np.testing.assert_array_equal(p1, p2)
