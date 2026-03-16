import gymnasium as gym
import numpy as np
import torch

from src.configs import CSTRConfig
from src.dataset import Normalizer
from src.dynamics_model import DynamicsEnsemble


class SetpointRewardWrapper(gym.Wrapper):
    """Replaces the environment's reward with setpoint tracking reward.

    This ensures both GT and learned envs use the same reward function.
    Also randomizes initial conditions for evaluation diversity.
    """

    def __init__(
        self,
        env: gym.Env,
        setpoints: dict[str, float],
        state_names: list[str],
        x0: np.ndarray | None = None,
        state_low: np.ndarray | None = None,
        state_high: np.ndarray | None = None,
    ) -> None:
        super().__init__(env)
        self.setpoints = setpoints
        self.state_names = state_names
        self._x0 = x0
        self._state_low = state_low
        self._state_high = state_high

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Perturb initial state if bounds are provided
        if self._x0 is not None and self._state_low is not None:
            noise_scale = 0.05 * (self._state_high - self._state_low)
            noise = self.np_random.uniform(-noise_scale, noise_scale).astype(np.float32)
            obs = np.clip(
                self._x0.astype(np.float32) + noise,
                self._state_low,
                self._state_high,
            ).astype(np.float32)
            # Overwrite the env's internal state
            if hasattr(self.env, 'state'):
                self.env.state = obs.copy()
            if hasattr(self.env, 'obs'):
                self.env.obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        reward = 0.0
        for var_name, sp_value in self.setpoints.items():
            idx = self.state_names.index(var_name)
            reward -= float((obs[idx] - sp_value) ** 2)
        return obs, reward, terminated, truncated, info


class LearnedCSTREnv(gym.Env):
    """Gymnasium environment that uses a learned dynamics ensemble for state transitions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ensemble: DynamicsEnsemble,
        state_normalizer: Normalizer,
        action_normalizer: Normalizer,
        delta_normalizer: Normalizer,
        config: CSTRConfig,
    ) -> None:
        super().__init__()
        self.ensemble = ensemble
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.delta_normalizer = delta_normalizer
        self.config = config

        self.observation_space = gym.spaces.Box(
            low=config.state_low.astype(np.float32),
            high=config.state_high.astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=config.action_low.astype(np.float32),
            high=config.action_high.astype(np.float32),
            dtype=np.float32,
        )

        self._state: np.ndarray | None = None
        self._step_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Randomize initial state around x0 (+/- 5% of range)
        noise_scale = 0.05 * (self.config.state_high - self.config.state_low)
        noise = self.np_random.uniform(-noise_scale, noise_scale).astype(np.float32)
        self._state = np.clip(
            self.config.x0.astype(np.float32) + noise,
            self.config.state_low,
            self.config.state_high,
        ).astype(np.float32)
        self._step_count = 0
        return self._state.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Must call reset() before step()"

        action = np.asarray(action, dtype=np.float32).reshape(self.config.action_dim)

        # Normalize inputs
        norm_state = self.state_normalizer.transform(
            self._state.reshape(1, -1)
        )
        norm_action = self.action_normalizer.transform(action.reshape(1, -1))

        # Run through ensemble
        state_t = torch.from_numpy(norm_state)
        action_t = torch.from_numpy(norm_action)
        prediction = self.ensemble.predict(state_t, action_t)

        # Denormalize predicted delta and apply
        norm_delta = prediction.mean.numpy()
        delta = self.delta_normalizer.inverse_transform(norm_delta).flatten()

        self._state = self._state + delta

        # Clamp to physical bounds
        self._state = np.clip(
            self._state, self.config.state_low, self.config.state_high
        ).astype(np.float32)

        self._step_count += 1

        # Compute reward (negative squared setpoint tracking error)
        reward = 0.0
        for var_name, sp_value in self.config.setpoints.items():
            idx = self.config.state_names.index(var_name)
            reward -= float((self._state[idx] - sp_value) ** 2)

        terminated = self._step_count >= self.config.episode_length
        truncated = False

        info = {
            "epistemic_uncertainty": prediction.epistemic_var.numpy().flatten(),
            "aleatoric_uncertainty": prediction.aleatoric_var.numpy().flatten(),
        }

        return self._state.copy(), reward, terminated, truncated, info
