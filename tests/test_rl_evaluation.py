import numpy as np
import pytest

from src.configs import CSTRConfig, EnsembleConfig, RLConfig
from src.dataset import Normalizer
from src.dynamics_model import DynamicsEnsemble
from src.learned_env import LearnedCSTREnv
from src.rl_evaluation import EvalResult, evaluate_agent, train_agent


@pytest.fixture
def tiny_learned_env() -> LearnedCSTREnv:
    config = CSTRConfig(episode_length=20)
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


class TestTrainAgent:
    def test_returns_trained_model(self, tiny_learned_env: LearnedCSTREnv) -> None:
        rl_config = RLConfig(total_timesteps=200)
        agent = train_agent(tiny_learned_env, rl_config)
        assert agent is not None
        assert hasattr(agent, "predict")


class TestEvaluateAgent:
    def test_returns_eval_result(self, tiny_learned_env: LearnedCSTREnv) -> None:
        rl_config = RLConfig(total_timesteps=200, eval_episodes=2)
        agent = train_agent(tiny_learned_env, rl_config)
        result = evaluate_agent(agent, tiny_learned_env, n_episodes=2)
        assert isinstance(result, EvalResult)
        assert isinstance(result.mean_reward, float)
        assert isinstance(result.std_reward, float)
        assert len(result.trajectories) == 2

    def test_trajectories_have_correct_shape(
        self, tiny_learned_env: LearnedCSTREnv
    ) -> None:
        rl_config = RLConfig(total_timesteps=200, eval_episodes=2)
        agent = train_agent(tiny_learned_env, rl_config)
        result = evaluate_agent(agent, tiny_learned_env, n_episodes=1)
        traj = result.trajectories[0]
        assert "states" in traj
        assert "actions" in traj
        assert "rewards" in traj
        # States should have shape (T, state_dim)
        assert traj["states"].ndim == 2
        assert traj["states"].shape[1] == 2
