from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.configs import RLConfig


@dataclass
class EvalResult:
    """Result from agent evaluation."""

    mean_reward: float = 0.0
    std_reward: float = 0.0
    trajectories: list[dict[str, np.ndarray]] = field(default_factory=list)


@dataclass
class SimToRealResult:
    """Result from sim-to-real comparison."""

    gt_in_gt: EvalResult = field(default_factory=EvalResult)
    learned_in_learned: EvalResult = field(default_factory=EvalResult)
    learned_in_gt: EvalResult = field(default_factory=EvalResult)


def train_agent(env: gym.Env, config: RLConfig) -> BaseAlgorithm:
    """Train an RL agent in the given environment."""
    algorithms = {"SAC": SAC, "PPO": PPO}
    algo_cls = algorithms[config.algorithm]
    agent = algo_cls("MlpPolicy", env, verbose=0)
    agent.learn(total_timesteps=config.total_timesteps)
    return agent


def evaluate_agent(
    agent: BaseAlgorithm, env: gym.Env, n_episodes: int = 10
) -> EvalResult:
    """Evaluate an agent and collect trajectory data."""
    episode_rewards: list[float] = []
    trajectories: list[dict[str, np.ndarray]] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        states = [obs.copy()]
        actions_list: list[np.ndarray] = []
        rewards_list: list[float] = []
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            states.append(obs.copy())
            actions_list.append(np.asarray(action).copy())
            rewards_list.append(float(reward))
            done = terminated or truncated

        episode_rewards.append(sum(rewards_list))
        trajectories.append(
            {
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions_list, dtype=np.float32),
                "rewards": np.array(rewards_list, dtype=np.float32),
            }
        )

    return EvalResult(
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        trajectories=trajectories,
    )


def sim_to_real_comparison(
    learned_env: gym.Env,
    gt_env: gym.Env,
    config: RLConfig,
) -> SimToRealResult:
    """Run the 3-way sim-to-real comparison.

    1. Train in GT, eval in GT
    2. Train in learned, eval in learned
    3. Train in learned, eval in GT
    """
    # 1. Train in ground truth, evaluate in ground truth
    gt_agent = train_agent(gt_env, config)
    gt_in_gt = evaluate_agent(gt_agent, gt_env, config.eval_episodes)

    # 2. Train in learned env, evaluate in learned env
    learned_agent = train_agent(learned_env, config)
    learned_in_learned = evaluate_agent(
        learned_agent, learned_env, config.eval_episodes
    )

    # 3. Evaluate learned-env agent in ground truth
    learned_in_gt = evaluate_agent(learned_agent, gt_env, config.eval_episodes)

    return SimToRealResult(
        gt_in_gt=gt_in_gt,
        learned_in_learned=learned_in_learned,
        learned_in_gt=learned_in_gt,
    )
