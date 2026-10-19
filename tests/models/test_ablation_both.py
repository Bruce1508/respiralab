from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from respiralab.models import ablation_both
from respiralab.models.ablation_both import (
    class_weight_tensor,
    focal_loss,
    mixup_pair,
    train_probe,
)


def test_class_weight_tensor_is_inverse_frequency():
    y_idx = np.array([0, 0, 0, 1])  # class 0: 3x as common as class 1
    weights = class_weight_tensor(y_idx, n_classes=2)
    assert weights[1] > weights[0]
    assert weights[1].item() == pytest.approx(weights[0].item() * 3, rel=0.01)


def test_focal_loss_with_gamma_zero_matches_cross_entropy():
    logits = torch.randn(5, 4)
    targets = torch.tensor([0, 1, 2, 3, 0])
    focal = focal_loss(logits, targets, gamma=0.0)
    ce = F.cross_entropy(logits, targets)
    assert focal.item() == pytest.approx(ce.item(), rel=1e-5)


def test_mixup_pair_rejects_non_train_patient():
    X = torch.randn(3, 4)
    y_onehot = F.one_hot(torch.tensor([0, 1, 2]), 4).float()
    patient_ids = np.array(["p1", "p2", "p_not_in_train"])
    train_patients = {"p1", "p2"}
    with pytest.raises(AssertionError, match="outside the train-fold patient set"):
        mixup_pair(X, y_onehot, patient_ids, train_patients)


def test_mixup_pair_accepts_all_train_patients_and_preserves_shape():
    X = torch.randn(3, 4)
    y_onehot = F.one_hot(torch.tensor([0, 1, 2]), 4).float()
    patient_ids = np.array(["p1", "p2", "p1"])
    train_patients = {"p1", "p2"}
    X_mixed, y_mixed = mixup_pair(X, y_onehot, patient_ids, train_patients)
    assert X_mixed.shape == X.shape
    assert y_mixed.shape == y_onehot.shape


@pytest.mark.parametrize("method", ["reference", "balanced_sampling", "focal_loss", "mixup"])
def test_train_probe_runs_end_to_end_on_synthetic_embeddings(method, monkeypatch):
    monkeypatch.setattr(ablation_both, "EMBEDDING_DIM", 8)
    rng = np.random.default_rng(0)
    n_train, n_val, dim = 40, 10, 8
    X_train = rng.normal(size=(n_train, dim)).astype(np.float32)
    y_train = rng.integers(0, 4, size=n_train)
    patient_ids_train = np.array([f"p{i % 5}" for i in range(n_train)])
    X_val = rng.normal(size=(n_val, dim)).astype(np.float32)
    y_val = rng.integers(0, 4, size=n_val)

    result = train_probe(
        X_train,
        y_train,
        patient_ids_train,
        X_val,
        y_val,
        method=method,
        max_epochs=3,
        patience=2,
    )

    assert 0.0 <= result["best_val_macro_f1"] <= 1.0
    assert result["n_epochs_run"] >= 1
    assert "both" in result["per_class"]
