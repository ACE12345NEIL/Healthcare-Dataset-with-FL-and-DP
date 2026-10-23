"""Neural Network and Machine Learning models for Federated Learning.

Includes architectures from the literature and research paper:
"Privacy-Preserving Federated Learning with Federated GANs for Early Heart Disease Prediction" (ICICC 2026):
1. LogisticRegressionNP: Baseline linear classifier for federated clinical benchmarks
2. MLPNP: Two-layer multilayer perceptron with ReLU activation
3. DeepResidualAttentionNP: Deep neural network combining feature self-attention,
   residual skip connections, and dropout (Paper Section 3.3)
4. SevenModelEnsembleNP: 7-model weighted ensemble aggregating predictions with
   weights proportional to squared accuracy (Paper Section 3.4 & Fig. 3)
"""
import numpy as np
from scipy.special import expit

EPS = 1e-12


class _NumpyModel:
    params: list

    def get_params(self) -> list:
        return [p.copy() for p in self.params]

    def set_params(self, params: list) -> None:
        if [p.shape for p in params] != [p.shape for p in self.params]:
            raise ValueError("Parameter shapes do not match the model architecture.")
        self.params = [p.copy() for p in params]

    def predict_proba(self, X) -> np.ndarray:
        return self._forward(X, training=False)[0]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def loss(self, X, y) -> float:
        p = np.clip(self.predict_proba(X), EPS, 1 - EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit_local(self, X, y, epochs: int, lr: float, batch_size: int, rng) -> None:
        """Mini-batch SGD on the given local patient data."""
        n = len(y)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                grads = self._gradients(X[idx], y[idx], rng=rng)
                for p, g in zip(self.params, grads):
                    p -= lr * g


class LogisticRegressionNP(_NumpyModel):
    """Logistic Regression baseline: params = [w (d,), b (1,)]"""

    def __init__(self, n_features: int, seed: int = 0):
        self.params = [np.zeros(n_features), np.zeros(1)]

    def _forward(self, X, training: bool = False):
        return expit(X @ self.params[0] + self.params[1][0]), None

    def _gradients(self, X, y, rng=None):
        p, _ = self._forward(X, training=True)
        err = (p - y) / len(y)
        return [X.T @ err, np.array([err.sum()])]


class MLPNP(_NumpyModel):
    """Two-layer MLP with ReLU activation.
    params = [W1 (d,h), b1 (h,), W2 (h,), b2 (1,)]
    """

    def __init__(self, n_features: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params = [
            rng.normal(0, np.sqrt(2.0 / n_features), (n_features, hidden)),
            np.zeros(hidden),
            rng.normal(0, np.sqrt(1.0 / hidden), hidden),
            np.zeros(1),
        ]

    def _forward(self, X, training: bool = False):
        W1, b1, W2, b2 = self.params
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)
        p = expit(a1 @ W2 + b2[0])
        return p, (z1, a1)

    def _gradients(self, X, y, rng=None):
        W1, b1, W2, b2 = self.params
        p, (z1, a1) = self._forward(X, training=True)
        dz2 = (p - y) / len(y)
        dW2 = a1.T @ dz2
        db2 = np.array([dz2.sum()])
        dz1 = (dz2[:, None] * W2[None, :]) * (z1 > 0)
        return [X.T @ dz1, dz1.sum(axis=0), dW2, db2]


class DeepResidualAttentionNP(_NumpyModel):
    """Deep Neural Network from ICICC 2026 Paper (Section 3.3):
    Combines:
    - Feature self-attention gate focusing on critical clinical biomarkers
    - Deep layers with ReLU activation
    - Residual skip connections: z2 = Relu(a1 W2 + b2) + a1
    - Dropout for regularization and preventing overfitting on hospital data
    
    params:
      0: w_att (d,) - feature attention weights
      1: b_att (d,) - feature attention bias
      2: W1 (d, hidden) - Dense 1 weights
      3: b1 (hidden,) - Dense 1 bias
      4: W2 (hidden, hidden) - Dense 2 weights
      5: b2 (hidden,) - Dense 2 bias
      6: W3 (hidden,) - Output layer weights
      7: b3 (1,) - Output bias
    """

    def __init__(self, n_features: int, hidden: int = 24, seed: int = 0, dropout: float = 0.05):
        self.n_features = n_features
        self.hidden = hidden
        self.dropout = dropout
        rng = np.random.default_rng(seed)
        self.params = [
            rng.normal(1.0, 0.1, n_features),                   # w_att
            np.zeros(n_features),                              # b_att
            rng.normal(0, np.sqrt(2.0 / n_features), (n_features, hidden)),  # W1
            np.zeros(hidden),                                  # b1
            rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden)),          # W2
            np.zeros(hidden),                                  # b2
            rng.normal(0, np.sqrt(1.0 / hidden), hidden),      # W3
            np.zeros(1),                                       # b3
        ]

    def _forward(self, X, training: bool = False, rng=None):
        w_att, b_att, W1, b1, W2, b2, W3, b3 = self.params
        # Feature attention gate:
        att_scores = expit(X * w_att + b_att)
        X_att = X * att_scores

        # Dense 1
        z1 = X_att @ W1 + b1
        a1 = np.maximum(z1, 0)

        # Dense 2 with Residual Connection (skip connection)
        z2 = a1 @ W2 + b2
        res2 = z2 + a1
        a2 = np.maximum(res2, 0)

        # Dropout
        if training and self.dropout > 0 and rng is not None:
            drop_mask = (rng.uniform(0, 1, a2.shape) >= self.dropout).astype(float) / (1.0 - self.dropout)
            a2_drop = a2 * drop_mask
        else:
            drop_mask = np.ones_like(a2)
            a2_drop = a2

        # Output projection
        logits = a2_drop @ W3 + b3[0]
        p = expit(logits)
        cache = (X, att_scores, X_att, z1, a1, z2, res2, a2, drop_mask, a2_drop, p)
        return p, cache

    def _gradients(self, X, y, rng=None):
        w_att, b_att, W1, b1, W2, b2, W3, b3 = self.params
        p, cache = self._forward(X, training=True, rng=rng)
        (X_raw, att_scores, X_att, z1, a1, z2, res2, a2, drop_mask, a2_drop, _) = cache
        n = len(y)
        dz_out = (p - y) / n

        # Output layer gradients
        dW3 = a2_drop.T @ dz_out
        db3 = np.array([dz_out.sum()])

        # Backprop through Dropout and Dense 2 + Residual
        da2_drop = dz_out[:, None] * W3[None, :]
        da2 = da2_drop * drop_mask
        dres2 = da2 * (res2 > 0)

        dW2 = a1.T @ dres2
        db2 = dres2.sum(axis=0)

        # Residual skip gradient: d/da1 (z2 + a1) = dres2 @ W2.T + dres2
        da1 = dres2 @ W2.T + dres2
        dz1 = da1 * (z1 > 0)

        dW1 = X_att.T @ dz1
        db1 = dz1.sum(axis=0)

        # Backprop through Feature Attention
        dX_att = dz1 @ W1.T
        # X_att = X_raw * att_scores -> d(att_scores) = dX_att * X_raw
        datt = dX_att * X_raw
        # att_scores = expit(S) where S = X_raw * w_att + b_att
        dS = datt * att_scores * (1.0 - att_scores)
        dw_att = (dS * X_raw).sum(axis=0)
        db_att = dS.sum(axis=0)

        return [dw_att, db_att, dW1, db1, dW2, db2, dW3, db3]


class SevenModelEnsembleNP:
    """7-Model Weighted Ensemble from ICICC 2026 Paper (Section 3.4 & Fig. 3):
    Trains seven diverse models with varied capacities and regularization,
    aggregating predictions with weights proportional to squared validation accuracy.
    """

    def __init__(self, n_features: int, seed: int = 42):
        self.n_features = n_features
        self.models = [
            LogisticRegressionNP(n_features, seed=seed),
            MLPNP(n_features, hidden=12, seed=seed + 1),
            MLPNP(n_features, hidden=16, seed=seed + 2),
            MLPNP(n_features, hidden=24, seed=seed + 3),
            DeepResidualAttentionNP(n_features, hidden=16, seed=seed + 4, dropout=0.03),
            DeepResidualAttentionNP(n_features, hidden=24, seed=seed + 5, dropout=0.05),
            DeepResidualAttentionNP(n_features, hidden=32, seed=seed + 6, dropout=0.08),
        ]
        self.weights = np.ones(7) / 7.0

    def fit_and_calibrate(self, X_train, y_train, X_val, y_val, epochs: int, lr: float, batch_size: int, rng):
        accuracies = []
        for i, m in enumerate(self.models):
            sub_rng = np.random.default_rng(rng.integers(0, 1000000))
            m.fit_local(X_train, y_train, epochs=epochs, lr=lr, batch_size=batch_size, rng=sub_rng)
            preds = m.predict(X_val)
            acc = float(np.mean(preds == y_val))
            accuracies.append(max(acc, 0.5))

        # Weight proportional to squared accuracy (Paper Fig. 3)
        sq_acc = np.array(accuracies) ** 2
        self.weights = sq_acc / sq_acc.sum()

    def predict_proba(self, X) -> np.ndarray:
        probas = np.array([m.predict_proba(X) for m in self.models])
        return np.tensordot(self.weights, probas, axes=(0, 0))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def make_model(name: str, n_features: int, seed: int = 0, hidden: int = 16):
    """Build a freshly initialised model; same (name, n_features, seed, hidden) -> same params."""
    if name == "logreg":
        return LogisticRegressionNP(n_features, seed)
    if name == "mlp":
        return MLPNP(n_features, hidden, seed)
    if name in ("deep_resnet", "attention_mlp", "resnet"):
        return DeepResidualAttentionNP(n_features, hidden=hidden, seed=seed)
    raise ValueError(f"Unknown model '{name}'. Use 'logreg', 'mlp', or 'deep_resnet'.")
