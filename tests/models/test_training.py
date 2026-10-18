from __future__ import annotations

import torch

from respiralab.models.cnn import CompactCNN
from respiralab.models.training import fit, load_checkpoint, should_stop_early


def _batch(seed: int, batch_size: int = 8) -> dict:
    torch.manual_seed(seed)
    return {
        "log_mel": torch.randn(batch_size, 64, 376),
        "mask": torch.ones(batch_size, 376, dtype=torch.bool),
        "label": torch.randint(0, 4, (batch_size,)),
    }


def test_should_stop_early_resets_on_improvement_and_stops_after_patience():
    best, count, stop = should_stop_early(0.5, 0.6, epochs_without_improvement=3, patience=2)
    assert (best, count, stop) == (0.6, 0, False)

    best, count, stop = should_stop_early(0.6, 0.55, epochs_without_improvement=0, patience=2)
    assert (best, count, stop) == (0.6, 1, False)

    best, count, stop = should_stop_early(0.6, 0.55, epochs_without_improvement=1, patience=2)
    assert (best, count, stop) == (0.6, 2, True)


def test_fit_selects_best_checkpoint_by_validation_metric_not_train_loss_or_last_epoch():
    torch.manual_seed(0)
    model = CompactCNN(n_classes=4)
    train_loader = [_batch(seed=1)]
    val_loader = [_batch(seed=2)]
    canned_val_scores = iter([{"macro_f1": 0.3}, {"macro_f1": 0.9}, {"macro_f1": 0.5}, {"macro_f1": 0.4}])

    def fake_evaluate(_model, _loader):
        scores = next(canned_val_scores)
        return {**scores, "loss": 0.0}

    result = fit(
        model,
        train_loader,
        val_loader,
        max_epochs=4,
        lr=1e-3,
        patience=10,
        evaluate_fn=fake_evaluate,
    )

    assert result["best_val_macro_f1"] == 0.9
    improved_flags = [h["improved"] for h in result["history"]]
    assert improved_flags == [True, True, False, False]


def test_resumed_run_reproduces_uninterrupted_run_epoch_by_epoch(tmp_path):
    train_loader = [_batch(seed=10)]
    val_loader = [_batch(seed=11)]

    torch.manual_seed(42)
    model_full = CompactCNN(n_classes=4)
    uninterrupted = fit(
        model_full,
        train_loader,
        val_loader,
        max_epochs=4,
        lr=1e-2,
        patience=10,
        checkpoint_path=tmp_path / "full.pt",
    )

    torch.manual_seed(42)
    model_a = CompactCNN(n_classes=4)
    first_half_ckpt = tmp_path / "half.pt"
    fit(
        model_a,
        train_loader,
        val_loader,
        max_epochs=2,
        lr=1e-2,
        patience=10,
        checkpoint_path=first_half_ckpt,
    )

    resume_state = load_checkpoint(first_half_ckpt)
    model_b = CompactCNN(n_classes=4)
    resumed = fit(
        model_b,
        train_loader,
        val_loader,
        max_epochs=4,
        lr=1e-2,
        patience=10,
        checkpoint_path=tmp_path / "resumed.pt",
        resume_state=resume_state,
    )

    assert len(resumed["history"]) == 4
    for expected, actual in zip(uninterrupted["history"], resumed["history"], strict=True):
        assert actual["epoch"] == expected["epoch"]
        assert actual["train_loss"] == expected["train_loss"]
        assert actual["val_macro_f1"] == expected["val_macro_f1"]
