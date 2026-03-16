from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.rl_evaluation import EvalResult, SimToRealResult


def set_publication_style() -> None:
    """Set matplotlib rcParams for clean, publication-quality figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4),
            "figure.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def plot_rollout_comparison(
    predicted: np.ndarray,
    actual: np.ndarray,
    uncertainty: np.ndarray | None,
    variable_names: list[str],
    save_path: Path,
    title: str = "Predicted vs Actual CSTR Trajectories",
) -> None:
    """Plot predicted vs actual trajectories with optional uncertainty bands.

    Args:
        predicted: (T, n_vars) predicted state trajectory
        actual: (T, n_vars) ground truth trajectory
        uncertainty: (T, n_vars) std dev for uncertainty bands, or None
        variable_names: names of state variables
        save_path: where to save the figure
    """
    set_publication_style()
    n_vars = predicted.shape[1]
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]

    t = np.arange(len(predicted))

    for i, (ax, name) in enumerate(zip(axes, variable_names)):
        ax.plot(t, actual[:, i], "b-", label="Ground Truth", alpha=0.8)
        ax.plot(t, predicted[:, i], "r--", label="Learned Model", alpha=0.8)
        if uncertainty is not None:
            ax.fill_between(
                t,
                predicted[:, i] - 2 * uncertainty[:, i],
                predicted[:, i] + 2 * uncertainty[:, i],
                color="red",
                alpha=0.15,
                label="95% CI",
            )
        ax.set_xlabel("Timestep")
        ax.set_ylabel(name)
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_multistep_error(
    errors_by_horizon: dict[int, dict[str, float]],
    variable_names: list[str],
    save_path: Path,
) -> None:
    """Plot MSE vs rollout horizon for each state variable.

    Args:
        errors_by_horizon: {horizon_length: {var_name: mse}}
        variable_names: names of state variables
        save_path: where to save
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    horizons = sorted(errors_by_horizon.keys())
    for name in variable_names:
        mses = [errors_by_horizon[h][name] for h in horizons]
        ax.plot(horizons, mses, "o-", label=name)

    ax.set_xlabel("Rollout Horizon (steps)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Multi-Step Prediction Error vs Horizon")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_sim_to_real(results: SimToRealResult, save_path: Path) -> None:
    """Bar chart comparing 3 evaluation conditions."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [
        "GT-trained\nin GT",
        "Learned-trained\nin Learned",
        "Learned-trained\nin GT",
    ]
    means = [
        results.gt_in_gt.mean_reward,
        results.learned_in_learned.mean_reward,
        results.learned_in_gt.mean_reward,
    ]
    stds = [
        results.gt_in_gt.std_reward,
        results.learned_in_learned.std_reward,
        results.learned_in_gt.std_reward,
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Sim-to-Real Transfer Gap")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_uncertainty_bands(
    trajectory: np.ndarray,
    epistemic_std: np.ndarray,
    aleatoric_std: np.ndarray,
    actual: np.ndarray,
    variable_names: list[str],
    save_path: Path,
) -> None:
    """Plot trajectory with separate epistemic and aleatoric uncertainty bands."""
    set_publication_style()
    n_vars = trajectory.shape[1]
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]

    t = np.arange(len(trajectory))

    for i, (ax, name) in enumerate(zip(axes, variable_names)):
        ax.plot(t, actual[:, i], "b-", label="Ground Truth", alpha=0.8)
        ax.plot(t, trajectory[:, i], "r--", label="Learned Model", alpha=0.8)

        total_std = np.sqrt(epistemic_std[:, i] ** 2 + aleatoric_std[:, i] ** 2)
        ax.fill_between(
            t,
            trajectory[:, i] - 2 * total_std,
            trajectory[:, i] + 2 * total_std,
            color="red",
            alpha=0.1,
            label="Total uncertainty",
        )
        ax.fill_between(
            t,
            trajectory[:, i] - 2 * epistemic_std[:, i],
            trajectory[:, i] + 2 * epistemic_std[:, i],
            color="orange",
            alpha=0.2,
            label="Epistemic uncertainty",
        )

        ax.set_xlabel("Timestep")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)

    fig.suptitle("Trajectory with Uncertainty Decomposition")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_training_curves(
    loss_curves: list[list[float]], save_path: Path
) -> None:
    """Plot training loss curves for all ensemble members."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, curve in enumerate(loss_curves):
        ax.plot(curve, label=f"Network {i + 1}", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NLL")
    ax.set_title("Ensemble Training Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
