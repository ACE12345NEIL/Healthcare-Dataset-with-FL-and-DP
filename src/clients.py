"""Simulated clients (hospitals). Each client owns only its own local partition."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Client:
    client_id: int
    X: np.ndarray
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def local_update(self, model_factory, global_params: list, epochs: int,
                     lr: float, batch_size: int, rng) -> list:
        """Start from the global params, train on local data, return the update
        (local params - global params). Raw data never leaves the client."""
        model = model_factory()
        model.set_params(global_params)
        model.fit_local(self.X, self.y, epochs, lr, batch_size, rng)
        return [l - g for l, g in zip(model.get_params(), global_params)]


def partition_data(X, y, n_clients: int, seed: int = 42, min_samples: int = 10) -> list:
    """Stratified random (IID) split of the TRAINING data into disjoint client partitions."""
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1.")
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_clients)]
    for label in np.unique(y):
        idx = rng.permutation(np.where(y == label)[0])
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            buckets[k].extend(chunk.tolist())

    clients = []
    for k, idx in enumerate(buckets):
        if len(idx) < min_samples:
            raise ValueError(f"Client {k} has only {len(idx)} samples (< {min_samples}). "
                             "Use fewer clients.")
        idx = np.array(sorted(idx))
        Xc, yc = X[idx].copy(), y[idx].copy()
        Xc.setflags(write=False)   # clients cannot be mutated by the server code
        yc.setflags(write=False)
        clients.append(Client(k, Xc, yc))
    return clients
