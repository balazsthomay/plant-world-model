"""Run the world model pipeline on Tennessee Eastman Process data."""

from pathlib import Path

import numpy as np
import torch

from src.configs import EnsembleConfig, TEPConfig
from src.data_collection import load_tep_data
from src.dataset import Normalizer, create_datasets
from src.dynamics_model import DynamicsEnsemble
from src.figures import (
    plot_multistep_error,
    plot_rollout_comparison,
    plot_training_curves,
    plot_uncertainty_bands,
)
from src.training import train_ensemble

FIGURES_DIR = Path("figures")
DATA_DIR = Path("data")
TEP_CSV = DATA_DIR / "tep" / "python_data_1year.csv"


def step1_load_and_analyze() -> dict[str, np.ndarray]:
    """Load TEP data and analyze action variation."""
    print("Step 1: Loading TEP data...")
    data = load_tep_data(TEP_CSV, normal_only=True, max_rows=50000)
    print(f"  Loaded {len(data['states'])} transitions")
    print(f"  State dim: {data['states'].shape[1]}, Action dim: {data['actions'].shape[1]}")

    # Analyze action variation
    print("\n  Action variation (coefficient of variation):")
    tep_config = TEPConfig()
    for i, name in enumerate(tep_config.action_names):
        vals = data["actions"][:, i]
        cv = vals.std() / (abs(vals.mean()) + 1e-10) * 100
        print(f"    {name:12s}  cv={cv:.1f}%  range=[{vals.min():.1f}, {vals.max():.1f}]")

    return data


def step2_train_model(
    data: dict[str, np.ndarray], tep_config: TEPConfig, ensemble_config: EnsembleConfig
) -> tuple:
    """Train dynamics ensemble on TEP data."""
    print("\nStep 2: Training dynamics ensemble on TEP data...")
    train_ds, val_ds, state_norm, action_norm, delta_norm = create_datasets(
        data, train_ratio=0.8
    )

    ensemble = DynamicsEnsemble(
        config=ensemble_config,
        state_dim=tep_config.state_dim,
        action_dim=tep_config.action_dim,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    result = train_ensemble(ensemble, train_ds, val_ds, ensemble_config, device=device)
    ensemble.to(torch.device("cpu"))

    plot_training_curves(result.loss_curves, FIGURES_DIR / "tep_training_curves.png")
    final_losses = [f"{c[-1]:.4f}" for c in result.loss_curves]
    print(f"  Training complete. Final val losses: {final_losses}")

    return ensemble, state_norm, action_norm, delta_norm


def step3_evaluate(
    ensemble: DynamicsEnsemble,
    state_norm: Normalizer,
    action_norm: Normalizer,
    delta_norm: Normalizer,
    data: dict[str, np.ndarray],
    tep_config: TEPConfig,
) -> None:
    """Evaluate model: one-step accuracy, multi-step rollouts."""
    print("\nStep 3: Evaluating TEP dynamics model...")

    # One-step prediction accuracy on held-out data
    n_test = 1000
    test_states = data["states"][-n_test:]
    test_actions = data["actions"][-n_test:]
    test_next = data["next_states"][-n_test:]

    norm_s = torch.from_numpy(state_norm.transform(test_states))
    norm_a = torch.from_numpy(action_norm.transform(test_actions))
    pred = ensemble.predict(norm_s, norm_a)
    pred_deltas = delta_norm.inverse_transform(pred.mean.numpy())
    pred_next = test_states + pred_deltas
    actual_next = test_next

    # Per-variable one-step MSE
    mse_per_var = np.mean((pred_next - actual_next) ** 2, axis=0)
    print("\n  One-step MSE per variable (top 5 worst):")
    worst_idx = np.argsort(mse_per_var)[-5:][::-1]
    for idx in worst_idx:
        name = tep_config.state_names[idx] if idx < len(tep_config.state_names) else f"Var{idx}"
        print(f"    {name:12s}  MSE={mse_per_var[idx]:.6f}")

    overall_mse = mse_per_var.mean()
    print(f"\n  Overall one-step MSE: {overall_mse:.6f}")

    # Multi-step rollout from a test point
    start_idx = len(data["states"]) - 200
    gt_states = data["states"][start_idx : start_idx + 100]
    gt_actions = data["actions"][start_idx : start_idx + 100]

    predicted_states = [gt_states[0].copy()]
    epistemic_stds = []
    aleatoric_stds = []

    current = gt_states[0].copy()
    for t in range(min(99, len(gt_actions) - 1)):
        norm_s = state_norm.transform(current.reshape(1, -1))
        norm_a = action_norm.transform(gt_actions[t : t + 1])
        pred = ensemble.predict(
            torch.from_numpy(norm_s), torch.from_numpy(norm_a)
        )
        delta = delta_norm.inverse_transform(pred.mean.numpy()).flatten()
        current = current + delta
        predicted_states.append(current.copy())

        ep_std = np.sqrt(pred.epistemic_var.numpy().flatten()) * delta_norm.std
        al_std = np.sqrt(pred.aleatoric_var.numpy().flatten()) * delta_norm.std
        epistemic_stds.append(ep_std)
        aleatoric_stds.append(al_std)

    predicted = np.array(predicted_states, dtype=np.float32)
    actual = gt_states[: len(predicted)]

    # Plot rollout for select key variables:
    # XMEAS(7)=reactor pressure, XMEAS(9)=reactor temp, XMEAS(17)=stripper underflow, XMEAS(20)=compressor work
    key_vars = [6, 8, 16, 19]  # 0-indexed
    key_names = [tep_config.state_names[i] for i in key_vars]

    ep_stds = np.array(epistemic_stds, dtype=np.float32)
    al_stds = np.array(aleatoric_stds, dtype=np.float32)
    cum_std = np.sqrt(np.cumsum(ep_stds**2 + al_stds**2, axis=0))
    cum_std_padded = np.vstack([np.zeros((1, tep_config.state_dim), dtype=np.float32), cum_std])

    plot_rollout_comparison(
        predicted[:, key_vars],
        actual[:, key_vars],
        cum_std_padded[:len(predicted), :][:, key_vars],
        key_names,
        FIGURES_DIR / "tep_rollout_comparison.png",
        title="Predicted vs Actual TEP Trajectories (Key Variables)",
    )

    # Multi-step error at different horizons
    horizons = [1, 5, 10, 25, 50]
    errors_by_horizon: dict[int, dict[str, float]] = {}
    for h in horizons:
        if h < len(predicted):
            errors = {}
            for vi, name in zip(key_vars, key_names):
                mse = float(np.mean((predicted[:h, vi] - actual[:h, vi]) ** 2))
                errors[name] = mse
            errors_by_horizon[h] = errors

    if errors_by_horizon:
        plot_multistep_error(
            errors_by_horizon,
            key_names,
            FIGURES_DIR / "tep_multistep_error.png",
        )

    # Uncertainty decomposition for key vars
    cum_ep_padded = np.vstack([np.zeros((1, tep_config.state_dim), dtype=np.float32), np.sqrt(np.cumsum(ep_stds**2, axis=0))])
    cum_al_padded = np.vstack([np.zeros((1, tep_config.state_dim), dtype=np.float32), np.sqrt(np.cumsum(al_stds**2, axis=0))])

    plot_uncertainty_bands(
        predicted[:, key_vars],
        cum_ep_padded[:len(predicted), :][:, key_vars],
        cum_al_padded[:len(predicted), :][:, key_vars],
        actual[:, key_vars],
        key_names,
        FIGURES_DIR / "tep_uncertainty.png",
    )

    print("  TEP evaluation complete. Figures saved.")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    tep_config = TEPConfig()
    ensemble_config = EnsembleConfig(
        n_networks=5,
        hidden_sizes=[256, 256, 256, 256],
        max_epochs=100,
        patience=10,
        batch_size=512,
    )

    data = step1_load_and_analyze()
    ensemble, state_norm, action_norm, delta_norm = step2_train_model(
        data, tep_config, ensemble_config
    )
    step3_evaluate(ensemble, state_norm, action_norm, delta_norm, data, tep_config)

    print("\nTEP pipeline complete! Figures saved to figures/")


if __name__ == "__main__":
    main()
