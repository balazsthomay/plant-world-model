from pathlib import Path

import numpy as np
import pytest

from src.figures import (
    plot_multistep_error,
    plot_rollout_comparison,
    plot_sim_to_real,
    plot_training_curves,
    plot_uncertainty_bands,
)
from src.rl_evaluation import EvalResult, SimToRealResult


@pytest.fixture
def tmp_fig_path(tmp_path: Path) -> Path:
    return tmp_path / "test_fig.png"


class TestPlotRolloutComparison:
    def test_creates_file(self, tmp_fig_path: Path) -> None:
        predicted = np.random.randn(50, 2).astype(np.float32)
        actual = np.random.randn(50, 2).astype(np.float32)
        uncertainty = np.abs(np.random.randn(50, 2)).astype(np.float32)
        plot_rollout_comparison(
            predicted, actual, uncertainty, ["Ca", "T"], tmp_fig_path
        )
        assert tmp_fig_path.exists()

    def test_without_uncertainty(self, tmp_fig_path: Path) -> None:
        predicted = np.random.randn(50, 2).astype(np.float32)
        actual = np.random.randn(50, 2).astype(np.float32)
        plot_rollout_comparison(predicted, actual, None, ["Ca", "T"], tmp_fig_path)
        assert tmp_fig_path.exists()


class TestPlotMultistepError:
    def test_creates_file(self, tmp_fig_path: Path) -> None:
        errors = {5: {"Ca": 0.01, "T": 0.1}, 10: {"Ca": 0.05, "T": 0.5}}
        plot_multistep_error(errors, ["Ca", "T"], tmp_fig_path)
        assert tmp_fig_path.exists()


class TestPlotSimToReal:
    def test_creates_file(self, tmp_fig_path: Path) -> None:
        results = SimToRealResult(
            gt_in_gt=EvalResult(mean_reward=-0.5, std_reward=0.1),
            learned_in_learned=EvalResult(mean_reward=-0.6, std_reward=0.1),
            learned_in_gt=EvalResult(mean_reward=-0.55, std_reward=0.1),
        )
        plot_sim_to_real(results, tmp_fig_path)
        assert tmp_fig_path.exists()


class TestPlotUncertaintyBands:
    def test_creates_file(self, tmp_fig_path: Path) -> None:
        n = 30
        traj = np.random.randn(n, 2).astype(np.float32)
        ep_std = np.abs(np.random.randn(n, 2)).astype(np.float32)
        al_std = np.abs(np.random.randn(n, 2)).astype(np.float32)
        actual = np.random.randn(n, 2).astype(np.float32)
        plot_uncertainty_bands(traj, ep_std, al_std, actual, ["Ca", "T"], tmp_fig_path)
        assert tmp_fig_path.exists()


class TestPlotTrainingCurves:
    def test_creates_file(self, tmp_fig_path: Path) -> None:
        curves = [[1.0, 0.8, 0.6, 0.5], [1.1, 0.9, 0.7, 0.55]]
        plot_training_curves(curves, tmp_fig_path)
        assert tmp_fig_path.exists()
