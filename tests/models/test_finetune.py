from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn

from respiralab.models import finetune
from respiralab.models.finetune import ASTClassifier, RawCycleDataset, run_epoch


class _FakeEncoder(nn.Module):
    """Stands in for the real AST encoder: same child names, tiny dims."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.layernorm = nn.LayerNorm(4)

    def forward(self, input_values: torch.Tensor, **_) -> SimpleNamespace:
        x = input_values.mean(dim=1)
        for layer in self.layers:
            x = layer(x)
        pooled = self.layernorm(x)
        return SimpleNamespace(pooler_output=pooled)


def _patched_classifier(monkeypatch, unfreeze_last_n: int) -> ASTClassifier:
    monkeypatch.setattr(finetune, "EMBEDDING_DIM", 4)
    monkeypatch.setattr(finetune.ASTModel, "from_pretrained", staticmethod(lambda _: _FakeEncoder()))
    return ASTClassifier(unfreeze_last_n=unfreeze_last_n, n_classes=4)


def test_unfreezes_only_last_n_layers_plus_final_layernorm_and_head(monkeypatch):
    model = _patched_classifier(monkeypatch, unfreeze_last_n=2)

    assert all(not p.requires_grad for p in model.encoder.layers[0].parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[1].parameters())
    assert all(p.requires_grad for p in model.encoder.layers[2].parameters())
    assert all(p.requires_grad for p in model.encoder.layers[3].parameters())
    assert all(p.requires_grad for p in model.encoder.layernorm.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_run_epoch_eval_mode_does_not_update_weights(monkeypatch):
    model = _patched_classifier(monkeypatch, unfreeze_last_n=1)
    before = model.classifier.weight.clone()

    loader = [({"input_values": torch.randn(3, 5, 4)}, torch.tensor([0, 1, 2]))]
    scores = run_epoch(model, loader, torch.device("cpu"), optimizer=None)

    assert torch.equal(before, model.classifier.weight)
    assert "macro_f1" in scores
    assert "loss" in scores


def test_run_epoch_training_mode_updates_only_unfrozen_weights(monkeypatch):
    model = _patched_classifier(monkeypatch, unfreeze_last_n=1)
    frozen_before = model.encoder.layers[0].weight.clone()
    unfrozen_before = model.encoder.layers[-1].weight.clone()
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.1)

    loader = [({"input_values": torch.randn(4, 5, 4)}, torch.tensor([0, 1, 2, 3]))]
    run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)

    assert torch.equal(frozen_before, model.encoder.layers[0].weight)
    assert not torch.equal(unfrozen_before, model.encoder.layers[-1].weight)


def test_raw_cycle_dataset_labels_and_shape(tmp_path):
    sr = 4000
    sf.write(str(tmp_path / "r1.wav"), np.zeros(sr * 2, dtype=np.float32), sr)
    rows = pd.DataFrame([{"record_id": "r1", "start_s": 0.0, "end_s": 1.0, "label4": "crackle"}])

    dataset = RawCycleDataset(rows, tmp_path)
    waveform, label_idx = dataset[0]

    assert label_idx == finetune.LABEL_TO_INDEX["crackle"]
    assert waveform.shape == (int(finetune.ENCODER_SAMPLE_RATE * 6.0),)
