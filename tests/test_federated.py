import numpy as np
import pytest

from src.clients import partition_data
from src.fedavg import apply_update, run_fedavg, weighted_average
from src.models import make_model
from src.preprocessing import prepare_data
from tests.synth import make_df


@pytest.fixture(scope="module")
def data():
    return prepare_data(make_df(500), seed=2)


# ---------- partitioning ----------
def test_partitions_disjoint_complete_stratified(data):
    X, y = data.X_train, data.y_train
    clients = partition_data(X, y, 5, seed=1)
    assert sum(c.n_samples for c in clients) == len(y)
    rows = [tuple(r) for c in clients for r in c.X]
    assert len(set(rows)) == len(rows) == len(y)          # no row appears twice
    rates = [c.y.mean() for c in clients]
    assert max(rates) - min(rates) < 0.1


def test_partition_errors_and_readonly(data):
    with pytest.raises(ValueError):
        partition_data(data.X_train, data.y_train, 0)
    with pytest.raises(ValueError):
        partition_data(data.X_train, data.y_train, 100)   # too few samples each
    c = partition_data(data.X_train, data.y_train, 3)[0]
    with pytest.raises(ValueError):
        c.X[0, 0] = 99.0


# ---------- models ----------
@pytest.mark.parametrize("name", ["logreg", "mlp"])
def test_gradients_match_finite_differences(name):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 4)), rng.integers(0, 2, 20)
    m = make_model(name, 4, seed=1, hidden=5)
    for p in m.params:
        p += rng.normal(0, 0.3, p.shape)        # move off the symmetric init
    grads = m._gradients(X, y)
    base = m.get_params()
    for i, g in enumerate(grads):
        flat = base[i].ravel()
        for j in range(min(4, flat.size)):
            eps = 1e-6
            plus, minus = [p.copy() for p in base], [p.copy() for p in base]
            plus[i].ravel()[j] += eps
            minus[i].ravel()[j] -= eps
            m.set_params(plus); lp = m.loss(X, y)
            m.set_params(minus); lm = m.loss(X, y)
            assert abs((lp - lm) / (2 * eps) - g.ravel()[j]) < 1e-5


def test_same_seed_same_init_and_shape_check():
    a, b = make_model("mlp", 8, seed=3), make_model("mlp", 8, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a.params, b.params))
    with pytest.raises(ValueError):
        a.set_params(make_model("mlp", 8, seed=3, hidden=4).params)


# ---------- aggregation ----------
def test_weighted_average_by_sample_count():
    out = weighted_average([[np.array([1., 1.])], [np.array([4., 4.])]], [1, 3])
    np.testing.assert_allclose(out[0], [3.25, 3.25])


def test_zero_updates_leave_global_unchanged():
    g = [np.ones(3), np.zeros(1)]
    new = apply_update(g, weighted_average([[np.zeros(3), np.zeros(1)]] * 2, [5, 7]))
    assert all(np.array_equal(a, b) for a, b in zip(g, new))


# ---------- FedAvg ----------
@pytest.mark.parametrize("name", ["logreg", "mlp"])
def test_fedavg_learns_and_is_reproducible(data, name):
    clients = partition_data(data.X_train, data.y_train, 5, seed=1)
    kw = dict(rounds=15, local_epochs=3, lr=0.1, seed=4)
    r1, p1 = run_fedavg(name, clients, data.X_test, data.y_test, **kw)
    r2, p2 = run_fedavg(name, clients, data.X_test, data.y_test, **kw)
    assert len(r1["history"]) == 15
    assert r1["metrics"]["roc_auc"] > 0.65
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))


def test_single_client_fedavg_equals_local_training(data):
    clients = partition_data(data.X_train, data.y_train, 1, seed=1)
    _, params = run_fedavg("logreg", clients, data.X_test, data.y_test,
                           rounds=1, local_epochs=2, lr=0.1, batch_size=32, seed=9)
    m = make_model("logreg", data.X_train.shape[1], 9)
    m.fit_local(clients[0].X, clients[0].y, 2, 0.1, 32, np.random.default_rng([9, 1, 0]))
    for a, b in zip(params, m.get_params()):
        np.testing.assert_allclose(a, b)
