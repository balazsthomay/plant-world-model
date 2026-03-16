import numpy as np
import pytest

from src.configs import CSTRConfig
from src.data_collection import collect_cstr_rollouts


class TestCollectCSTRRollouts:
    def test_returns_required_keys(self, cstr_config: CSTRConfig) -> None:
        data = collect_cstr_rollouts(
            n_episodes=1, steps_per_episode=10, config=cstr_config, seed=42
        )
        assert "states" in data
        assert "actions" in data
        assert "next_states" in data

    def test_correct_shapes(self, cstr_config: CSTRConfig) -> None:
        n_episodes = 2
        steps = 10
        data = collect_cstr_rollouts(
            n_episodes=n_episodes, steps_per_episode=steps, config=cstr_config, seed=42
        )
        total = n_episodes * steps
        assert data["states"].shape == (total, cstr_config.state_dim)
        assert data["actions"].shape == (total, cstr_config.action_dim)
        assert data["next_states"].shape == (total, cstr_config.state_dim)

    def test_states_are_finite(self, cstr_config: CSTRConfig) -> None:
        data = collect_cstr_rollouts(
            n_episodes=1, steps_per_episode=50, config=cstr_config, seed=42
        )
        assert np.all(np.isfinite(data["states"]))
        assert np.all(np.isfinite(data["actions"]))
        assert np.all(np.isfinite(data["next_states"]))

    def test_actions_within_bounds(self, cstr_config: CSTRConfig) -> None:
        data = collect_cstr_rollouts(
            n_episodes=2, steps_per_episode=50, config=cstr_config, seed=42
        )
        assert np.all(data["actions"] >= cstr_config.action_low)
        assert np.all(data["actions"] <= cstr_config.action_high)

    def test_different_seeds_give_different_data(self, cstr_config: CSTRConfig) -> None:
        data1 = collect_cstr_rollouts(
            n_episodes=1, steps_per_episode=20, config=cstr_config, seed=42
        )
        data2 = collect_cstr_rollouts(
            n_episodes=1, steps_per_episode=20, config=cstr_config, seed=99
        )
        assert not np.allclose(data1["actions"], data2["actions"])

    def test_action_strategies_produce_different_distributions(
        self, cstr_config: CSTRConfig
    ) -> None:
        data_random = collect_cstr_rollouts(
            n_episodes=1,
            steps_per_episode=50,
            config=cstr_config,
            seed=42,
            action_strategy="random",
        )
        data_sinusoidal = collect_cstr_rollouts(
            n_episodes=1,
            steps_per_episode=50,
            config=cstr_config,
            seed=42,
            action_strategy="sinusoidal",
        )
        assert not np.allclose(data_random["actions"], data_sinusoidal["actions"])

    def test_float32_dtype(self, cstr_config: CSTRConfig) -> None:
        data = collect_cstr_rollouts(
            n_episodes=1, steps_per_episode=10, config=cstr_config, seed=42
        )
        assert data["states"].dtype == np.float32
        assert data["actions"].dtype == np.float32
        assert data["next_states"].dtype == np.float32
