from __future__ import annotations

import torch

from respiralab.models.cnn import CompactCNN, train_smoke_batch


def _random_batch(batch_size: int, n_mel: int = 64, n_frames: int = 376) -> dict:
    torch.manual_seed(0)
    return {
        "log_mel": torch.randn(batch_size, n_mel, n_frames),
        "mask": torch.ones(batch_size, n_frames, dtype=torch.bool),
        "label": torch.randint(0, 4, (batch_size,)),
    }


def test_forward_output_shape_matches_batch_and_n_classes():
    model = CompactCNN(n_classes=4)
    batch = _random_batch(batch_size=5)
    logits = model(batch["log_mel"], batch["mask"])
    assert logits.shape == (5, 4)


def test_masked_pooling_ignores_padded_frames():
    model = CompactCNN(n_classes=4)
    model.eval()
    torch.manual_seed(1)
    real = torch.randn(1, 64, 200)
    pad_a = torch.zeros(1, 64, 176)
    pad_b = torch.full((1, 64, 176), 999.0)
    log_mel_a = torch.cat([real, pad_a], dim=2)
    log_mel_b = torch.cat([real, pad_b], dim=2)
    mask = torch.zeros(1, 376, dtype=torch.bool)
    mask[:, :200] = True

    with torch.no_grad():
        logits_a = model(log_mel_a, mask)
        logits_b = model(log_mel_b, mask)

    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_train_smoke_batch_overfits_tiny_batch():
    model = CompactCNN(n_classes=4)
    batch = _random_batch(batch_size=8)
    losses = train_smoke_batch(model, batch, n_steps=100, lr=1e-2, seed=0)

    assert len(losses) == 100
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1
