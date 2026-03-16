from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.configs import EnsembleConfig
from src.dataset import TransitionDataset
from src.dynamics_model import DynamicsEnsemble


@dataclass
class TrainingResult:
    """Results from ensemble training."""

    loss_curves: list[list[float]] = field(default_factory=list)


def _create_bootstrap_sampler(
    dataset_size: int, rng: torch.Generator
) -> SubsetRandomSampler:
    """Create a bootstrap sampler (sampling with replacement)."""
    indices = torch.randint(0, dataset_size, (dataset_size,), generator=rng).tolist()
    return SubsetRandomSampler(indices)


def _evaluate(
    network: torch.nn.Module, val_loader: DataLoader, device: torch.device
) -> float:
    """Compute mean validation loss."""
    total_loss = 0.0
    n_batches = 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            loss = network.loss(inputs, targets)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_ensemble(
    ensemble: DynamicsEnsemble,
    train_dataset: TransitionDataset,
    val_dataset: TransitionDataset,
    config: EnsembleConfig,
    device: torch.device | None = None,
) -> TrainingResult:
    """Train each network in the ensemble independently with bootstrap sampling."""
    if device is None:
        device = torch.device("cpu")

    ensemble.to(device)
    result = TrainingResult()

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    for i, network in enumerate(ensemble.networks):
        rng = torch.Generator().manual_seed(i)

        if config.bootstrap:
            sampler = _create_bootstrap_sampler(len(train_dataset), rng)
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True
            )

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        loss_curve: list[float] = []

        for epoch in range(config.max_epochs):
            # Training
            network.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = network.loss(inputs, targets)
                loss.backward()
                optimizer.step()

            # Validation
            network.eval()
            val_loss = _evaluate(network, val_loader, device)
            loss_curve.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        result.loss_curves.append(loss_curve)

    return result
