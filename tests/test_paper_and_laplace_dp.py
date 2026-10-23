import numpy as np
import pytest

from src.clients import partition_data
from src.fedavg import run_fedavg
from src.models import DeepResidualAttentionNP, SevenModelEnsembleNP, make_model
from src.preprocessing import enrich_clinical_features, prepare_data
from src.privacy import (
    add_laplace_noise,
    compute_dp_guarantee,
    laplace_scale_from_epsilon,
    privatize_update,
    update_l1_norm,
)
from tests.synth import make_df


def test_laplace_noise_properties():
    # Laplace(0, b) mean should be 0, std should be sqrt(2) * b
    b = 2.0
    u = [np.zeros(250_000)]
    rng = np.random.default_rng(42)
    noisy = add_laplace_noise(u, b, rng)
    assert abs(noisy[0].mean()) < 0.03
    expected_std = np.sqrt(2.0) * b
    assert noisy[0].std() == pytest.approx(expected_std, rel=0.03)


def test_laplace_dp_calibration():
    clip_norm = 0.2
    epsilon = 1.0  # as recommended in ICICC 2026 paper
    scale = laplace_scale_from_epsilon(clip_norm, epsilon)
    assert scale == pytest.approx(0.2)

    # Test privatize_update with laplace mechanism
    u = [np.array([1.0, -1.0]), np.array([2.0])]
    rng = np.random.default_rng(123)
    noisy, norm = privatize_update(u, clip_norm=clip_norm, mechanism="laplace", epsilon=epsilon, rng=rng)
    assert norm == pytest.approx(4.0)  # L1 norm = 1 + 1 + 2 = 4
    assert len(noisy) == len(u)


def test_compute_dp_guarantee():
    res_laplace = compute_dp_guarantee("laplace", clip_norm=0.2, noise_scale=0.2, rounds=10)
    assert res_laplace["provable"] is True
    assert res_laplace["epsilon"] > 0
    assert "Laplace" in res_laplace["mechanism"]

    res_gaussian = compute_dp_guarantee("gaussian", clip_norm=0.2, noise_scale=0.1, rounds=10)
    assert res_gaussian["provable"] is True
    assert res_gaussian["epsilon"] > 0
    assert "Gaussian" in res_gaussian["mechanism"]


def test_deep_residual_attention_gradient_check():
    """Verify analytical gradients match finite differences for DeepResidualAttentionNP."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 4))
    y = rng.integers(0, 2, 10)
    model = DeepResidualAttentionNP(n_features=4, hidden=6, seed=12, dropout=0.0)

    # Move parameters off zero/symmetric init
    for p in model.params:
        p += rng.normal(0, 0.1, p.shape)

    grads = model._gradients(X, y)
    base_params = model.get_params()

    # Check first few entries of each parameter tensor
    for i, g in enumerate(grads):
        flat_p = base_params[i].ravel()
        flat_g = g.ravel()
        for j in range(min(3, flat_p.size)):
            eps = 1e-6
            plus = [p.copy() for p in base_params]
            minus = [p.copy() for p in base_params]
            plus[i].ravel()[j] += eps
            minus[i].ravel()[j] -= eps

            model.set_params(plus)
            lp = model.loss(X, y)
            model.set_params(minus)
            lm = model.loss(X, y)

            num_grad = (lp - lm) / (2.0 * eps)
            assert abs(num_grad - flat_g[j]) < 1e-4


def test_fedavg_with_deep_resnet_and_laplace_dp():
    df = make_df(300)
    data = prepare_data(df, seed=42)
    clients = partition_data(data.X_train, data.y_train, 3, seed=42)

    res, _ = run_fedavg(
        "deep_resnet", clients, data.X_test, data.y_test,
        rounds=5, local_epochs=2, lr=0.05, clip_norm=0.2,
        mechanism="laplace", epsilon=1.0, seed=42
    )

    assert res["model"] == "fedavg_deep_resnet"
    assert res["privacy_mode"] == "clip_laplace_noise"
    assert res["dp_guarantee"]["provable"] is True
    assert len(res["history"]) == 5
    assert "accuracy" in res["metrics"]


def test_seven_model_ensemble():
    df = make_df(200)
    data = prepare_data(df, seed=42)
    ensemble = SevenModelEnsembleNP(n_features=data.X_train.shape[1], seed=42)
    rng = np.random.default_rng(42)

    ensemble.fit_and_calibrate(
        data.X_train, data.y_train, data.X_test, data.y_test,
        epochs=3, lr=0.05, batch_size=32, rng=rng
    )

    preds = ensemble.predict(data.X_test)
    probas = ensemble.predict_proba(data.X_test)
    assert len(preds) == len(data.y_test)
    assert len(probas) == len(data.y_test)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    assert ensemble.weights.sum() == pytest.approx(1.0)


def test_clinical_feature_enrichment():
    df = make_df(50)
    enriched = enrich_clinical_features(df)
    assert "MetabolicRiskScore" in enriched.columns
    assert "InsulinResistanceProxy" in enriched.columns
    assert "Glucose_x_Age" in enriched.columns
    assert "HighGlucose_Flag" in enriched.columns
    assert len(enriched.columns) >= 20
