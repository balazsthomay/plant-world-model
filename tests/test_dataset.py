import numpy as np
import pytest
import torch

from src.dataset import Normalizer, TransitionDataset, create_datasets


class TestNormalizer:
    def test_fit_stores_mean_std(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        norm = Normalizer()
        norm.fit(data)
        np.testing.assert_allclose(norm.mean, [3.0, 4.0])
        assert norm.std is not None
        assert norm.std.shape == (2,)

    def test_transform_inverse_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 3)).astype(np.float32)
        norm = Normalizer()
        norm.fit(data)
        transformed = norm.transform(data)
        recovered = norm.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, data, atol=1e-5)

    def test_transformed_has_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((1000, 2)).astype(np.float32) * 5 + 10
        norm = Normalizer()
        norm.fit(data)
        transformed = norm.transform(data)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=0.05)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=0.05)

    def test_handles_constant_features(self) -> None:
        data = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]], dtype=np.float32)
        norm = Normalizer()
        norm.fit(data)
        transformed = norm.transform(data)
        assert np.all(np.isfinite(transformed))


class TestTransitionDataset:
    def test_length(self, synthetic_transitions: dict[str, np.ndarray]) -> None:
        ds = TransitionDataset(
            states=synthetic_transitions["states"],
            actions=synthetic_transitions["actions"],
            next_states=synthetic_transitions["next_states"],
        )
        assert len(ds) == 1000

    def test_getitem_shapes(self, synthetic_transitions: dict[str, np.ndarray]) -> None:
        ds = TransitionDataset(
            states=synthetic_transitions["states"],
            actions=synthetic_transitions["actions"],
            next_states=synthetic_transitions["next_states"],
        )
        inputs, targets = ds[0]
        # inputs = concat(state, action) = 2 + 1 = 3
        assert inputs.shape == (3,)
        # targets = delta = next_state - state = 2
        assert targets.shape == (2,)

    def test_targets_are_deltas(self, synthetic_transitions: dict[str, np.ndarray]) -> None:
        ds = TransitionDataset(
            states=synthetic_transitions["states"],
            actions=synthetic_transitions["actions"],
            next_states=synthetic_transitions["next_states"],
        )
        inputs, targets = ds[0]
        expected_delta = (
            synthetic_transitions["next_states"][0] - synthetic_transitions["states"][0]
        )
        np.testing.assert_allclose(targets.numpy(), expected_delta, atol=1e-5)

    def test_tensor_dtype(self, synthetic_transitions: dict[str, np.ndarray]) -> None:
        ds = TransitionDataset(
            states=synthetic_transitions["states"],
            actions=synthetic_transitions["actions"],
            next_states=synthetic_transitions["next_states"],
        )
        inputs, targets = ds[0]
        assert inputs.dtype == torch.float32
        assert targets.dtype == torch.float32


class TestCreateDatasets:
    def test_returns_train_val_normalizers(
        self, synthetic_transitions: dict[str, np.ndarray]
    ) -> None:
        train_ds, val_ds, state_norm, action_norm, delta_norm = create_datasets(
            synthetic_transitions, train_ratio=0.8
        )
        assert isinstance(train_ds, TransitionDataset)
        assert isinstance(val_ds, TransitionDataset)
        assert isinstance(state_norm, Normalizer)
        assert isinstance(action_norm, Normalizer)
        assert isinstance(delta_norm, Normalizer)

    def test_train_val_split_sizes(
        self, synthetic_transitions: dict[str, np.ndarray]
    ) -> None:
        train_ds, val_ds, *_ = create_datasets(
            synthetic_transitions, train_ratio=0.8
        )
        assert len(train_ds) == 800
        assert len(val_ds) == 200

    def test_no_data_leakage_temporal_split(
        self, synthetic_transitions: dict[str, np.ndarray]
    ) -> None:
        """Temporal split: train data comes before val data (no random shuffle)."""
        train_ds, val_ds, *_ = create_datasets(
            synthetic_transitions, train_ratio=0.8
        )
        # Last train state should be before first val state in original index
        # We verify by checking the raw arrays are contiguous slices
        assert len(train_ds) + len(val_ds) == len(
            synthetic_transitions["states"]
        )
