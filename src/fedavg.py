"""Manually controlled FedAvg (sample-count-weighted) in a single process,
with rigorous Differential Privacy via Laplace Mechanism (pure epsilon-DP)
or Gaussian Mechanism ((epsilon, delta)-DP)."""
import time

import numpy as np

from src.evaluation import METRIC_NAMES, compute_metrics
from src.models import make_model
from src.privacy import (
    compute_dp_guarantee,
    gaussian_std_from_epsilon,
    laplace_scale_from_epsilon,
    privatize_update,
    update_l2_norm,
)


def weighted_average(updates: list, n_samples: list) -> list:
    """updates: one list-of-arrays per client. Weights = n_k / sum(n)."""
    w = np.asarray(n_samples, dtype=float)
    w = w / w.sum()
    return [sum(wk * u[i] for wk, u in zip(w, updates)) for i in range(len(updates[0]))]


def apply_update(global_params: list, update: list) -> list:
    return [g + u for g, u in zip(global_params, update)]


def evaluate_params(model_factory, params, X_test, y_test) -> dict:
    model = model_factory()
    model.set_params(params)
    score = model.predict_proba(X_test)
    return compute_metrics(y_test, (score >= 0.5).astype(int), score)


def run_fedavg(model_name: str, clients: list, X_test, y_test, rounds: int = 30,
               local_epochs: int = 5, lr: float = 0.05, batch_size: int = 32,
               seed: int = 42, hidden: int = 16,
               clip_norm=None, noise_multiplier: float = 0.0,
               mechanism: str = "gaussian", epsilon: float = None, delta: float = 1e-5):
    """Run Federated Averaging with optional Differential Privacy.

    Privacy modes:
      clip_norm=None                          -> plain FedAvg (control)
      clip_norm=C, noise_multiplier=0, eps=None -> clipping only
      clip_norm=C, mechanism='gaussian'       -> clipping + Gaussian noise
      clip_norm=C, mechanism='laplace'        -> clipping + Laplace noise (pure epsilon-DP)

    The server sees only privatized model updates and sample counts, never raw patient records.
    """
    if rounds < 1 or local_epochs < 1:
        raise ValueError("rounds and local_epochs must be >= 1.")
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be >= 0.")
    if (noise_multiplier > 0 or epsilon is not None) and clip_norm is None:
        raise ValueError("Differential privacy noise requires a clip_norm to bound sensitivity.")

    n_features = clients[0].X.shape[1]

    def factory():
        return make_model(model_name, n_features, seed, hidden)

    global_params = factory().get_params()      # shared initialisation
    n_samples = [c.n_samples for c in clients]
    history = []

    # Calculate calibrated noise scale with bounded sensitivity C
    if epsilon is not None and clip_norm is not None:
        if mechanism == "laplace":
            noise_scale = laplace_scale_from_epsilon(clip_norm, epsilon)
            noise_multiplier = noise_scale / clip_norm
        else:
            noise_scale = gaussian_std_from_epsilon(clip_norm, epsilon, delta)
            noise_multiplier = noise_scale / clip_norm
    elif clip_norm is not None:
        noise_scale = noise_multiplier * clip_norm
    else:
        noise_scale = 0.0

    t0 = time.perf_counter()
    for rnd in range(1, rounds + 1):
        updates = [c.local_update(factory, global_params, local_epochs, lr, batch_size,
                                  np.random.default_rng([seed, rnd, c.client_id]))
                   for c in clients]
        norms = [update_l2_norm(u) for u in updates]

        if clip_norm is not None:
            noisy_updates = []
            for u, c in zip(updates, clients):
                p_u, _ = privatize_update(
                    u, clip_norm=clip_norm, noise_multiplier=noise_multiplier,
                    rng=np.random.default_rng([seed, rnd, c.client_id, 1]),
                    mechanism=mechanism, epsilon=epsilon, delta=delta
                )
                noisy_updates.append(p_u)
            updates = noisy_updates

        global_params = apply_update(global_params, weighted_average(updates, n_samples))
        m = evaluate_params(factory, global_params, X_test, y_test)
        history.append({
            "round": rnd, **m,
            "mean_update_norm": float(np.mean(norms)),
            "frac_clipped": float(np.mean([n > clip_norm for n in norms])) if clip_norm else 0.0,
        })
    train_time = time.perf_counter() - t0

    has_noise = (noise_multiplier > 0) or (epsilon is not None)
    if clip_norm is None:
        mode = "none"
    elif not has_noise:
        mode = "clip_only"
    elif mechanism == "laplace":
        mode = "clip_laplace_noise"
    else:
        mode = "clip_gaussian_noise"

    # Compute formal DP certificate
    if has_noise and clip_norm is not None:
        dp_guarantee = compute_dp_guarantee(mechanism, clip_norm, noise_scale, rounds, delta)
    else:
        dp_guarantee = {
            "mechanism": "None",
            "epsilon": float("inf"),
            "delta": 0.0,
            "guarantee": "No differential privacy active"
        }

    result = {
        "model": f"fedavg_{model_name}",
        "n_clients": len(clients),
        "client_sizes": n_samples,
        "communication_rounds": rounds,
        "local_epochs": local_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "privacy_mode": mode,
        "mechanism": mechanism,
        "clip_norm": clip_norm,
        "noise_multiplier": noise_multiplier,
        "noise_scale": noise_scale,
        "noise_std_per_client": noise_scale,
        "dp_guarantee": dp_guarantee,
        "train_time_sec": train_time,
        "metrics": {k: history[-1][k] for k in METRIC_NAMES + ["confusion_matrix"]},
        "history": history,
    }
    return result, global_params
