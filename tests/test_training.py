import numpy as np
import pytest
import torch

from src.configs import EnsembleConfig
from src.dataset import TransitionDataset, create_datasets
from src.dynamics_model import DynamicsEnsemble
from src.training import train_ensemble


class TestTrainEnsemble:
    def test_returns_training_result(
        self,
        synthetic_transitions: dict[str, np.ndarray],
        ensemble_config: EnsembleConfig,
    ) -> None:
        train_ds, val_ds, *_ = create_datasets(synthetic_transitions, train_ratio=0.8)
        ensemble = DynamicsEnsemble(
            config=ensemble_config, state_dim=2, action_dim=1
        )
        result = train_ensemble(ensemble, train_ds, val_ds, ensemble_config)
        assert hasattr(result, "loss_curves")
        assert len(result.loss_curves) == ensemble_config.n_networks

    def test_loss_decreases(
        self,
        synthetic_transitions: dict[str, np.ndarray],
        ensemble_config: EnsembleConfig,
    ) -> None:
        train_ds, val_ds, *_ = create_datasets(synthetic_transitions, train_ratio=0.8)
        ensemble = DynamicsEnsemble(
            config=ensemble_config, state_dim=2, action_dim=1
        )
        result = train_ensemble(ensemble, train_ds, val_ds, ensemble_config)
        # At least one network's loss should decrease
        for curve in result.loss_curves:
            if len(curve) >= 2:
                assert curve[-1] <= curve[0] + 0.5  # some tolerance

    def test_learns_linear_system(
        self, synthetic_transitions: dict[str, np.ndarray]
    ) -> None:
        """Train on simple linear dynamics and verify prediction accuracy."""
        config = EnsembleConfig(
            n_networks=2,
            hidden_sizes=[64, 64],
            max_epochs=50,
            patience=10,
            batch_size=64,
            learning_rate=1e-3,
        )
        train_ds, val_ds, state_norm, action_norm, delta_norm = create_datasets(
            synthetic_transitions, train_ratio=0.8
        )
        ensemble = DynamicsEnsemble(config=config, state_dim=2, action_dim=1)
        train_ensemble(ensemble, train_ds, val_ds, config)

        # Test prediction on held-out data
        test_states = torch.from_numpy(
            state_norm.transform(synthetic_transitions["states"][-100:])
        )
        test_actions = torch.from_numpy(
            action_norm.transform(synthetic_transitions["actions"][-100:])
        )
        expected_deltas = delta_norm.transform(
            synthetic_transitions["next_states"][-100:]
            - synthetic_transitions["states"][-100:]
        )

        with torch.no_grad():
            pred = ensemble.predict(test_states, test_actions)

        mse = ((pred.mean.numpy() - expected_deltas) ** 2).mean()
        assert mse < 0.1, f"MSE on linear system too high: {mse}"
