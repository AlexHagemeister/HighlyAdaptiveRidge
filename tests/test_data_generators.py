import numpy as np
import pytest

from har.data_generators import (
    SmoothDataGenerator,
    JumpDataGenerator,
    SinusoidalDataGenerator,
)

DGPS = [SmoothDataGenerator, JumpDataGenerator, SinusoidalDataGenerator]
DIMS = [1, 3, 5]


class TestOutputShapes:
    @pytest.mark.parametrize("dgp", DGPS, ids=lambda d: d.name)
    @pytest.mark.parametrize("d", DIMS)
    def test_generate_data_shape(self, dgp, d):
        n = 25
        X, Y = dgp.generate_data(n, d, seed=0)
        assert X.shape == (n, d)
        assert Y.shape == (n,)


class TestSeedReproducibility:
    @pytest.mark.parametrize("dgp", DGPS, ids=lambda d: d.name)
    def test_same_seed_same_output(self, dgp):
        X1, Y1 = dgp.generate_data(20, 3, seed=42)
        X2, Y2 = dgp.generate_data(20, 3, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    @pytest.mark.parametrize("dgp", DGPS, ids=lambda d: d.name)
    def test_different_seed_different_output(self, dgp):
        X1, _ = dgp.generate_data(20, 3, seed=0)
        X2, _ = dgp.generate_data(20, 3, seed=1)
        assert not np.array_equal(X1, X2)


class TestNoiselessDeterminism:
    """Target functions are deterministic for d in {1, 3, 5}."""

    @pytest.mark.parametrize("dgp", DGPS, ids=lambda d: d.name)
    @pytest.mark.parametrize("d", DIMS)
    def test_target_deterministic(self, dgp, d):
        rng = np.random.default_rng(7)
        X = rng.uniform(-2, 2, (30, d))
        f = {1: dgp.f_1, 3: dgp.f_3, 5: dgp.f_5}[d]
        y1 = f(X)
        y2 = f(X)
        np.testing.assert_array_equal(y1, y2)
