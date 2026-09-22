import numpy as np
import pytest

from nnbench import metrics
from nnbench.models import ARCHITECTURES, build


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
@pytest.mark.parametrize("shape", [(32, 32, 3), (28, 28, 1)])
def test_architectures_build_and_predict(arch, shape):
    model = build(arch, shape, 10)
    probs = model.predict(np.random.rand(4, *shape).astype("float32"), verbose=0)
    assert probs.shape == (4, 10)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_metrics_perfect_predictions():
    y = np.array([0, 1, 2, 2])
    probs = np.eye(3)[y]
    m = metrics.classification_metrics(y, probs)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["ece"] == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    y = np.array([0, 1, 0, 1])
    probs = np.array([[0.99, 0.01]] * 4)  # always confident class 0, right half the time
    assert metrics.expected_calibration_error(y, probs) == pytest.approx(0.49, abs=1e-6)


def test_mcnemar():
    a = np.ones(100, bool)
    assert metrics.mcnemar_p(a, a) == 1.0
    b = a.copy()
    b[:30] = False
    assert metrics.mcnemar_p(a, b) < 1e-6


def test_bootstrap_ci_contains_point_estimate():
    correct = np.random.default_rng(0).random(1000) < 0.8
    lo, hi = metrics.bootstrap_accuracy_ci(correct)
    assert lo < correct.mean() < hi
