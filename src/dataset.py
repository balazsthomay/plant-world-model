import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Zero-mean, unit-variance normalizer with constant-feature handling."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        self.mean = data.mean(axis=0).astype(np.float32)
        std = data.std(axis=0).astype(np.float32)
        # Avoid division by zero for constant features
        std[std < 1e-8] = 1.0
        self.std = std
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Must call fit() before transform()"
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Must call fit() before inverse_transform()"
        return (data * self.std + self.mean).astype(np.float32)


class TransitionDataset(Dataset):
    """PyTorch dataset of (state, action) -> delta transitions."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        state_normalizer: Normalizer | None = None,
        action_normalizer: Normalizer | None = None,
        delta_normalizer: Normalizer | None = None,
    ) -> None:
        deltas = next_states - states

        if state_normalizer is not None:
            states = state_normalizer.transform(states)
        if action_normalizer is not None:
            actions = action_normalizer.transform(actions)
        if delta_normalizer is not None:
            deltas = delta_normalizer.transform(deltas)

        self.inputs = torch.from_numpy(
            np.concatenate([states, actions], axis=1).astype(np.float32)
        )
        self.targets = torch.from_numpy(deltas.astype(np.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_datasets(
    data: dict[str, np.ndarray],
    train_ratio: float = 0.8,
) -> tuple[TransitionDataset, TransitionDataset, Normalizer, Normalizer, Normalizer]:
    """Create normalized train/val datasets from transition data.

    Uses temporal split (no shuffle) to avoid data leakage.
    Returns (train_dataset, val_dataset, state_normalizer, action_normalizer, delta_normalizer).
    """
    n = len(data["states"])
    split = int(n * train_ratio)

    train_states = data["states"][:split]
    train_actions = data["actions"][:split]
    train_next = data["next_states"][:split]
    val_states = data["states"][split:]
    val_actions = data["actions"][split:]
    val_next = data["next_states"][split:]

    # Fit normalizers on training data only
    state_norm = Normalizer().fit(train_states)
    action_norm = Normalizer().fit(train_actions)
    delta_norm = Normalizer().fit(train_next - train_states)

    train_ds = TransitionDataset(
        train_states, train_actions, train_next, state_norm, action_norm, delta_norm
    )
    val_ds = TransitionDataset(
        val_states, val_actions, val_next, state_norm, action_norm, delta_norm
    )

    return train_ds, val_ds, state_norm, action_norm, delta_norm
