from dataclasses import dataclass, field

import numpy as np


@dataclass
class CSTRConfig:
    """Configuration for the CSTR environment."""

    state_dim: int = 2
    action_dim: int = 1
    state_names: list[str] = field(default_factory=lambda: ["Ca", "T"])
    action_names: list[str] = field(default_factory=lambda: ["Tc"])
    state_low: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 300.0])
    )
    state_high: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 400.0])
    )
    action_low: np.ndarray = field(default_factory=lambda: np.array([295.0]))
    action_high: np.ndarray = field(default_factory=lambda: np.array([302.0]))
    x0: np.ndarray = field(default_factory=lambda: np.array([0.87725294, 324.475443]))
    dt: float = 0.25
    episode_length: int = 100
    setpoints: dict[str, float] = field(
        default_factory=lambda: {"Ca": 0.85}
    )


@dataclass
class EnsembleConfig:
    """Configuration for the dynamics model ensemble."""

    n_networks: int = 5
    hidden_sizes: list[int] = field(
        default_factory=lambda: [200, 200, 200, 200]
    )
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 5
    bootstrap: bool = True


@dataclass
class RLConfig:
    """Configuration for RL training and evaluation."""

    algorithm: str = "SAC"
    total_timesteps: int = 50_000
    eval_episodes: int = 20
