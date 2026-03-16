import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from src.configs import CSTRConfig, EnsembleConfig
from src.dataset import Normalizer
from src.dynamics_model import DynamicsEnsemble
from src.learned_env import LearnedCSTREnv


@pytest.fixture
def learned_env() -> LearnedCSTREnv:
    """Create a LearnedCSTREnv with a small untrained ensemble."""
    config = CSTRConfig()
    ensemble_config = EnsembleConfig(n_networks=2, hidden_sizes=[16, 16])
    ensemble = DynamicsEnsemble(
        config=ensemble_config, state_dim=config.state_dim, action_dim=config.action_dim
    )

    state_norm = Normalizer()
    state_norm.mean = np.array([0.5, 330.0], dtype=np.float32)
    state_norm.std = np.array([0.2, 30.0], dtype=np.float32)

    action_norm = Normalizer()
    action_norm.mean = np.array([298.0], dtype=np.float32)
    action_norm.std = np.array([3.0], dtype=np.float32)

    delta_norm = Normalizer()
    delta_norm.mean = np.array([0.0, 0.0], dtype=np.float32)
    delta_norm.std = np.array([0.01, 1.0], dtype=np.float32)

    return LearnedCSTREnv(
        ensemble=ensemble,
        state_normalizer=state_norm,
        action_normalizer=action_norm,
        delta_normalizer=delta_norm,
        config=config,
    )


class TestLearnedCSTREnv:
    def test_gymnasium_check_env(self, learned_env: LearnedCSTREnv) -> None:
        check_env(learned_env, skip_render_check=True)

    def test_reset_returns_correct_types(self, learned_env: LearnedCSTREnv) -> None:
        obs, info = learned_env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.dtype == np.float32

    def test_reset_observation_shape(self, learned_env: LearnedCSTREnv) -> None:
        obs, _ = learned_env.reset()
        assert obs.shape == (2,)

    def test_step_returns_correct_types(self, learned_env: LearnedCSTREnv) -> None:
        learned_env.reset(seed=42)
        action = learned_env.action_space.sample()
        obs, reward, terminated, truncated, info = learned_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self, learned_env: LearnedCSTREnv) -> None:
        learned_env.reset(seed=42)
        action = learned_env.action_space.sample()
        obs, *_ = learned_env.step(action)
        assert obs.shape == (2,)

    def test_info_contains_uncertainty(self, learned_env: LearnedCSTREnv) -> None:
        learned_env.reset(seed=42)
        action = learned_env.action_space.sample()
        _, _, _, _, info = learned_env.step(action)
        assert "epistemic_uncertainty" in info
        assert "aleatoric_uncertainty" in info

    def test_terminates_at_max_steps(self, learned_env: LearnedCSTREnv) -> None:
        learned_env.reset(seed=42)
        terminated = False
        for _ in range(learned_env.config.episode_length + 10):
            action = learned_env.action_space.sample()
            _, _, terminated, truncated, _ = learned_env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_no_nan_in_rollout(self, learned_env: LearnedCSTREnv) -> None:
        learned_env.reset(seed=42)
        for _ in range(50):
            action = learned_env.action_space.sample()
            obs, _, terminated, truncated, _ = learned_env.step(action)
            assert np.all(np.isfinite(obs)), f"NaN or Inf in observation: {obs}"
            if terminated or truncated:
                break

    def test_reward_is_finite(self, learned_env: LearnedCSTREnv) -> None:
        learned_env.reset(seed=42)
        for _ in range(20):
            action = learned_env.action_space.sample()
            _, reward, terminated, truncated, _ = learned_env.step(action)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
