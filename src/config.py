"""Single reproducible experiment configuration (JSON-serialisable)."""
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_path: str = "data/diabetes.csv"
    out_dir: str = "results"
    seed: int = 42
    test_size: float = 0.2
    n_clients: int = 5
    rounds: int = 30
    local_epochs: int = 5
    lr: float = 0.05
    batch_size: int = 32
    hidden: int = 16
    rf_estimators: int = 200
    clip_norm: float = 0.2                 # L2 clipping bound C (typical client update norm is ~0.1-0.3)
    noise_levels: list = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
    compare_noise: float = 0.1             # level used in the "with vs without noise" comparison
    n_repeats: int = 3                     # repeats vary training/noise seed, not the split

    def validate(self) -> "ExperimentConfig":
        pos_int = ["n_clients", "rounds", "local_epochs", "batch_size", "hidden",
                   "rf_estimators", "n_repeats"]
        for k in pos_int:
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError(f"'{k}' must be an integer >= 1.")
        if not isinstance(self.seed, int):
            raise ValueError("'seed' must be an integer.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("'test_size' must be in (0, 1).")
        if self.lr <= 0:
            raise ValueError("'lr' must be > 0.")
        if self.clip_norm <= 0:
            raise ValueError("'clip_norm' must be > 0.")
        self.noise_levels = [float(x) for x in self.noise_levels]
        if not self.noise_levels or any(x <= 0 for x in self.noise_levels):
            raise ValueError("'noise_levels' must be a non-empty list of values > 0 "
                             "(zero noise is run automatically as 'clip only').")
        if float(self.compare_noise) not in self.noise_levels:
            raise ValueError("'compare_noise' must be one of 'noise_levels'.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d).validate()

    @classmethod
    def from_json(cls, path) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return cls.from_dict(json.loads(path.read_text()))
