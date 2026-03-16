from pathlib import Path

import numpy as np
import pcgym

from src.configs import CSTRConfig


def _make_cstr_env(config: CSTRConfig) -> object:
    """Create a PC-Gym CSTR environment from config."""
    env_params = {
        "model": "cstr",
        "x0": config.x0.copy(),
        "a_space": {"low": config.action_low, "high": config.action_high},
        "o_space": {"low": config.state_low, "high": config.state_high},
        "SP": {},
        "N": config.episode_length,
        "tsim": config.episode_length * config.dt,
        "normalise_a": False,
        "normalise_o": False,
        "integration_method": "casadi",
    }
    return pcgym.make_env(env_params)


def _generate_action(
    step: int,
    strategy: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a control action based on the chosen strategy."""
    mid = (action_low + action_high) / 2
    half_range = (action_high - action_low) / 2

    if strategy == "random":
        return rng.uniform(action_low, action_high).astype(np.float32)
    elif strategy == "sinusoidal":
        period = rng.uniform(10, 50)
        phase = rng.uniform(0, 2 * np.pi)
        return (mid + half_range * 0.8 * np.sin(2 * np.pi * step / period + phase)).astype(
            np.float32
        )
    elif strategy == "step":
        # Hold action for blocks of 10-30 steps
        if step % int(rng.uniform(10, 30)) == 0:
            return rng.uniform(action_low, action_high).astype(np.float32)
        return mid.astype(np.float32)
    elif strategy == "mixed":
        choice = rng.choice(["random", "sinusoidal", "step"])
        return _generate_action(step, choice, action_low, action_high, rng)
    else:
        raise ValueError(f"Unknown action strategy: {strategy}")


def collect_cstr_rollouts(
    n_episodes: int,
    steps_per_episode: int,
    config: CSTRConfig,
    seed: int = 42,
    action_strategy: str = "random",
) -> dict[str, np.ndarray]:
    """Collect (state, action, next_state) transitions from the CSTR environment.

    Returns dict with keys 'states', 'actions', 'next_states', each (N, dim) float32.
    """
    rng = np.random.default_rng(seed)
    env_config = CSTRConfig(
        **{**config.__dict__, "episode_length": steps_per_episode + 1}
    )
    env = _make_cstr_env(env_config)

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_next_states: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        state = obs.astype(np.float32)

        for step in range(steps_per_episode):
            action = _generate_action(
                step, action_strategy, config.action_low, config.action_high, rng
            )
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_state = next_obs.astype(np.float32)

            all_states.append(state.copy())
            all_actions.append(action.copy())
            all_next_states.append(next_state.copy())

            state = next_state
            if terminated or truncated:
                break

    return {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "next_states": np.array(all_next_states, dtype=np.float32),
    }


def load_tep_data(
    csv_path: Path,
    normal_only: bool = True,
    max_rows: int | None = None,
) -> dict[str, np.ndarray]:
    """Load Tennessee Eastman Process data from the anasouzac dataset.

    Format: semicolon-delimited, timestamp index, XMEAS(1-41) + XMV(1-11) + STATUS.
    Separates into 41 measured (state) and 11 manipulated (action) variables.

    Args:
        csv_path: Path to python_data_1year.csv or similar.
        normal_only: If True, filter to STATUS=0 (normal operation).
        max_rows: Limit number of rows to load (None for all).
    """
    import pandas as pd

    df = pd.read_csv(csv_path, sep=";", index_col=0, nrows=max_rows)

    if normal_only and "STATUS" in df.columns:
        df = df[df["STATUS"] == 0].reset_index(drop=True)

    state_cols = [c for c in df.columns if c.startswith("XMEAS")]
    action_cols = [c for c in df.columns if c.startswith("XMV")]

    states = df[state_cols].values.astype(np.float32)
    actions = df[action_cols].values.astype(np.float32)

    return {
        "states": states[:-1],
        "actions": actions[:-1],
        "next_states": states[1:],
    }
