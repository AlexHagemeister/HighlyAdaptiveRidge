import numpy as np
import pytest

from har.hal import HAL


@pytest.fixture
def small_data():
    """Small 1-d dataset for HAL (n=20, p=1)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (20, 1))
    Y = 2 * X[:, 0] + 1
    return X, Y


class TestFitPredict:
    def test_predict_shape(self, small_data):
        X, Y = small_data
        model = HAL()
        model.fit(X, Y)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_predict_before_fit(self):
        model = HAL()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([[0.5]]))


class TestBasisMatrix:
    def test_basis_dimensions(self):
        rng = np.random.default_rng(1)
        n, p = 10, 2
        X = rng.uniform(0, 1, (n, p))
        model = HAL()
        model.knots = X
        H = model._bases(X)
        # For p=2: n*(2^p - 1) = 10*3 = 30 basis columns
        assert H.shape[0] == n
        assert H.shape[1] == n * (2**p - 1)
