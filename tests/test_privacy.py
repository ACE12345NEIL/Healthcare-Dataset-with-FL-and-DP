import numpy as np
import pytest

from src.clients import partition_data
from src.fedavg import run_fedavg
from src.preprocessing import prepare_data
from src.privacy import add_gaussian_noise, clip_update, privatize_update, update_l2_norm
from tests.synth import make_df


def _u():
    return [np.array([3.0, 4.0]), np.array([0.0])]       # L2 norm = 5


def test_clip_bounds_norm_and_keeps_direction():
    clipped, norm = clip_update(_u(), 1.0)
    assert norm == pytest.approx(5.0)
    assert update_l2_norm(clipped) == pytest.approx(1.0)
    np.testing.assert_allclose(clipped[0], [0.6, 0.8])


def test_small_update_not_changed():
    clipped, _ = clip_update(_u(), 10.0)
    np.testing.assert_allclose(clipped[0], [3.0, 4.0])


def test_zero_noise_equals_clip_only():
    a, _ = privatize_update(_u(), 1.0, 0.0, np.random.default_rng(0))
    b, _ = clip_update(_u(), 1.0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_noise_std_matches_multiplier_times_clip():
    big = [np.zeros(200_000)]
    out, _ = privatize_update(big, clip_norm=2.0, noise_multiplier=0.5, rng=np.random.default_rng(1))
    assert out[0].std() == pytest.approx(1.0, rel=0.02)      # 0.5 * 2.0
    assert abs(out[0].mean()) < 0.02


def test_noise_reproducible_and_seed_dependent():
    a = add_gaussian_noise(_u(), 1.0, np.random.default_rng(5))
    b = add_gaussian_noise(_u(), 1.0, np.random.default_rng(5))
    c = add_gaussian_noise(_u(), 1.0, np.random.default_rng(6))
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])


def test_invalid_privacy_args():
    with pytest.raises(ValueError):
        clip_update(_u(), 0)
    with pytest.raises(ValueError):
        add_gaussian_noise(_u(), -1, np.random.default_rng(0))


@pytest.fixture(scope="module")
def setup():
    d = prepare_data(make_df(500), seed=2)
    return d, partition_data(d.X_train, d.y_train, 5, seed=1)


KW = dict(rounds=10, local_epochs=3, lr=0.1, seed=4)


def test_huge_clip_zero_noise_equals_control(setup):
    d, clients = setup
    _, p0 = run_fedavg("logreg", clients, d.X_test, d.y_test, **KW)
    _, p1 = run_fedavg("logreg", clients, d.X_test, d.y_test, clip_norm=1e9, **KW)
    assert all(np.allclose(a, b) for a, b in zip(p0, p1))


def test_noise_changes_result_and_is_reproducible(setup):
    d, clients = setup
    _, p0 = run_fedavg("logreg", clients, d.X_test, d.y_test, clip_norm=1.0, **KW)
    _, p1 = run_fedavg("logreg", clients, d.X_test, d.y_test, clip_norm=1.0, noise_multiplier=0.5, **KW)
    _, p2 = run_fedavg("logreg", clients, d.X_test, d.y_test, clip_norm=1.0, noise_multiplier=0.5, **KW)
    assert not all(np.allclose(a, b) for a, b in zip(p0, p1))
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))


def test_extreme_noise_hurts_and_requires_clip(setup):
    d, clients = setup
    ctrl, _ = run_fedavg("logreg", clients, d.X_test, d.y_test, **KW)
    noisy, _ = run_fedavg("logreg", clients, d.X_test, d.y_test, clip_norm=1.0,
                          noise_multiplier=100.0, **KW)
    assert noisy["metrics"]["roc_auc"] < ctrl["metrics"]["roc_auc"]
    assert noisy["privacy_mode"] == "clip_gaussian_noise" and ctrl["privacy_mode"] == "none"
    with pytest.raises(ValueError):
        run_fedavg("logreg", clients, d.X_test, d.y_test, noise_multiplier=0.5, **KW)
