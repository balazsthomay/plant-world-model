import numpy as np
import pytest
import torch

from src.configs import CSTRConfig, EnsembleConfig


@pytest.fixture
def cstr_config() -> CSTRConfig:
    return CSTRConfig()


@pytest.fixture
def ensemble_config() -> EnsembleConfig:
    return EnsembleConfig(
        n_networks=2,
        hidden_sizes=[32, 32],
        max_epochs=5,
        patience=2,
        batch_size=32,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_transitions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate synthetic (state, action, next_state) data from a simple linear system."""
    n_samples = 1000
    state_dim = 2
    action_dim = 1

    states = rng.standard_normal((n_samples, state_dim)).astype(np.float32)
    actions = rng.standard_normal((n_samples, action_dim)).astype(np.float32)

    # Simple linear dynamics: next_state = 0.9 * state + 0.1 * action_broadcast
    next_states = (
        0.9 * states + 0.1 * np.broadcast_to(actions, (n_samples, state_dim))
    ).astype(np.float32)

    return {
        "states": states,
        "actions": actions,
        "next_states": next_states,
    }


@pytest.fixture
def torch_device() -> torch.device:
    return torch.device("cpu")
