"""Run the full world model pipeline: data collection -> training -> evaluation -> figures."""

from pathlib import Path

import numpy as np
import torch

from src.configs import CSTRConfig, EnsembleConfig, RLConfig
from src.data_collection import collect_cstr_rollouts
from src.dataset import create_datasets
from src.dynamics_model import DynamicsEnsemble
from src.figures import (
    plot_multistep_error,
    plot_rollout_comparison,
    plot_sim_to_real,
    plot_training_curves,
    plot_uncertainty_bands,
)
from src.learned_env import LearnedCSTREnv, SetpointRewardWrapper
from src.rl_evaluation import evaluate_agent, sim_to_real_comparison, train_agent
from src.training import train_ensemble

FIGURES_DIR = Path("figures")
DATA_DIR = Path("data")


def setup_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def step1_collect_data(cstr_config: CSTRConfig) -> dict[str, np.ndarray]:
    """Collect diverse CSTR rollout data."""
    print("Step 1: Collecting CSTR rollout data...")
    all_data: dict[str, list[np.ndarray]] = {"states": [], "actions": [], "next_states": []}

    for strategy in ["random", "sinusoidal", "step"]:
        data = collect_cstr_rollouts(
            n_episodes=10,
            steps_per_episode=100,
            config=cstr_config,
            seed=42,
            action_strategy=strategy,
        )
        for key in all_data:
            all_data[key].append(data[key])

    combined = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    print(f"  Collected {len(combined['states'])} transitions")
    np.savez(DATA_DIR / "cstr_rollouts.npz", **combined)
    return combined


def step2_train_model(
    data: dict[str, np.ndarray], ensemble_config: EnsembleConfig
) -> tuple:
    """Train the dynamics ensemble."""
    print("Step 2: Training dynamics ensemble...")
    train_ds, val_ds, state_norm, action_norm, delta_norm = create_datasets(
        data, train_ratio=0.8
    )

    ensemble = DynamicsEnsemble(config=ensemble_config, state_dim=2, action_dim=1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    result = train_ensemble(ensemble, train_ds, val_ds, ensemble_config, device=device)

    # Move back to CPU for inference
    ensemble.to(torch.device("cpu"))

    # Save model
    ensemble.save(str(DATA_DIR / "ensemble.pt"))
    plot_training_curves(result.loss_curves, FIGURES_DIR / "training_curves.png")
    final_losses = [f"{c[-1]:.4f}" for c in result.loss_curves]
    print(f"  Training complete. Final val losses: {final_losses}")

    return ensemble, state_norm, action_norm, delta_norm


def step3_evaluate_model(
    ensemble: DynamicsEnsemble,
    state_norm,
    action_norm,
    delta_norm,
    data: dict[str, np.ndarray],
    cstr_config: CSTRConfig,
) -> None:
    """Evaluate the model: rollout comparison, multi-step error, uncertainty."""
    print("Step 3: Evaluating dynamics model...")

    # Multi-step rollout comparison
    from src.data_collection import _make_cstr_env

    env = _make_cstr_env(cstr_config)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)

    # Collect a ground truth trajectory
    gt_states = [obs.copy()]
    actions_taken = []
    for step in range(100):
        action = rng.uniform(cstr_config.action_low, cstr_config.action_high).astype(np.float32)
        obs, _, done, _, _ = env.step(action)
        gt_states.append(obs.copy())
        actions_taken.append(action)
        if done:
            break

    gt_states = np.array(gt_states, dtype=np.float32)
    actions_arr = np.array(actions_taken, dtype=np.float32)

    # Predict multi-step rollout from initial state
    horizon = min(len(gt_states), 100)
    predicted_states = [gt_states[0].copy()]
    epistemic_stds = []
    aleatoric_stds = []

    current = gt_states[0].copy()
    for t in range(horizon - 1):
        norm_s = state_norm.transform(current.reshape(1, -1))
        norm_a = action_norm.transform(actions_arr[t : t + 1])
        pred = ensemble.predict(
            torch.from_numpy(norm_s), torch.from_numpy(norm_a)
        )
        delta = delta_norm.inverse_transform(pred.mean.numpy()).flatten()
        current = current + delta
        current = np.clip(current, cstr_config.state_low, cstr_config.state_high)
        predicted_states.append(current.copy())

        # Convert normalized variance to real-space std
        ep_std = np.sqrt(pred.epistemic_var.numpy().flatten()) * delta_norm.std
        al_std = np.sqrt(pred.aleatoric_var.numpy().flatten()) * delta_norm.std
        epistemic_stds.append(ep_std)
        aleatoric_stds.append(al_std)

    predicted = np.array(predicted_states, dtype=np.float32)
    actual = gt_states[:len(predicted)]

    # Cumulative uncertainty grows over time
    ep_stds = np.array(epistemic_stds, dtype=np.float32)
    al_stds = np.array(aleatoric_stds, dtype=np.float32)
    cum_ep = np.cumsum(ep_stds, axis=0)
    cum_al = np.cumsum(al_stds, axis=0)

    # Plot rollout comparison
    total_std = np.sqrt(cum_ep**2 + cum_al**2)
    # Pad to match predicted length (first state has no uncertainty)
    total_std_padded = np.vstack([np.zeros((1, 2), dtype=np.float32), total_std])

    plot_rollout_comparison(
        predicted,
        actual,
        total_std_padded,
        cstr_config.state_names,
        FIGURES_DIR / "rollout_comparison.png",
    )

    # Uncertainty decomposition plot
    cum_ep_padded = np.vstack([np.zeros((1, 2), dtype=np.float32), cum_ep])
    cum_al_padded = np.vstack([np.zeros((1, 2), dtype=np.float32), cum_al])
    plot_uncertainty_bands(
        predicted,
        cum_ep_padded,
        cum_al_padded,
        actual,
        cstr_config.state_names,
        FIGURES_DIR / "uncertainty_decomposition.png",
    )

    # Multi-step error at different horizons
    horizons = [1, 5, 10, 25, 50]
    errors_by_horizon: dict[int, dict[str, float]] = {}
    for h in horizons:
        if h < len(predicted):
            mse_ca = float(np.mean((predicted[:h, 0] - actual[:h, 0]) ** 2))
            mse_t = float(np.mean((predicted[:h, 1] - actual[:h, 1]) ** 2))
            errors_by_horizon[h] = {"Ca": mse_ca, "T": mse_t}

    if errors_by_horizon:
        plot_multistep_error(
            errors_by_horizon,
            cstr_config.state_names,
            FIGURES_DIR / "multistep_error.png",
        )

    print("  Model evaluation complete. Figures saved.")


def step4_rl_evaluation(
    ensemble: DynamicsEnsemble,
    state_norm,
    action_norm,
    delta_norm,
    cstr_config: CSTRConfig,
    rl_config: RLConfig,
) -> None:
    """Train RL agents and evaluate sim-to-real transfer."""
    print("Step 4: RL training and sim-to-real evaluation...")

    from src.data_collection import _make_cstr_env

    learned_env = LearnedCSTREnv(
        ensemble=ensemble,
        state_normalizer=state_norm,
        action_normalizer=action_norm,
        delta_normalizer=delta_norm,
        config=cstr_config,
    )
    gt_env = SetpointRewardWrapper(
        _make_cstr_env(cstr_config),
        setpoints=cstr_config.setpoints,
        state_names=cstr_config.state_names,
    )

    results = sim_to_real_comparison(learned_env, gt_env, rl_config)

    print(f"  GT-trained in GT:       {results.gt_in_gt.mean_reward:.2f} +/- {results.gt_in_gt.std_reward:.2f}")
    print(f"  Learned-trained in Learned: {results.learned_in_learned.mean_reward:.2f} +/- {results.learned_in_learned.std_reward:.2f}")
    print(f"  Learned-trained in GT:  {results.learned_in_gt.mean_reward:.2f} +/- {results.learned_in_gt.std_reward:.2f}")

    plot_sim_to_real(results, FIGURES_DIR / "sim_to_real.png")
    print("  Sim-to-real evaluation complete.")


def main() -> None:
    setup_dirs()

    cstr_config = CSTRConfig()
    ensemble_config = EnsembleConfig(
        n_networks=5,
        hidden_sizes=[200, 200, 200, 200],
        max_epochs=100,
        patience=10,
        batch_size=256,
    )
    rl_config = RLConfig(total_timesteps=50_000, eval_episodes=10)

    # Pipeline
    data = step1_collect_data(cstr_config)
    ensemble, state_norm, action_norm, delta_norm = step2_train_model(data, ensemble_config)
    step3_evaluate_model(ensemble, state_norm, action_norm, delta_norm, data, cstr_config)
    step4_rl_evaluation(ensemble, state_norm, action_norm, delta_norm, cstr_config, rl_config)

    print("\nPipeline complete! Figures saved to figures/")


if __name__ == "__main__":
    main()
