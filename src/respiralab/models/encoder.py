"""D47: pretrained audio encoder for v2 (ADR-0001 D2/D4b, experiments E05+).

Checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram
Transformer, AudioSet-pretrained). License and input-rate verification:
docs/07_SOURCE_VERIFICATION.md item 6.

Embeddings are frozen (ASTModel.eval(), no_grad) pooled 768-d vectors --
exactly the "frozen encoder embeddings -> logistic regression" input E05
needs. Input must already be at CHECKPOINT_SAMPLE_RATE (16 kHz), i.e. the
output of dsp.encoder_input.prepare_encoder_input.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTModel

from respiralab.dsp.encoder_input import ENCODER_SAMPLE_RATE

CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHECKPOINT_SAMPLE_RATE = 16000
EMBEDDING_DIM = 768


class EncoderInputError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_encoder() -> tuple[ASTModel, ASTFeatureExtractor]:
    model = ASTModel.from_pretrained(CHECKPOINT)
    model.eval()
    feature_extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    if feature_extractor.sampling_rate != CHECKPOINT_SAMPLE_RATE:
        raise EncoderInputError(
            f"checkpoint {CHECKPOINT} expects {feature_extractor.sampling_rate}Hz, "
            f"not the assumed {CHECKPOINT_SAMPLE_RATE}Hz -- amend ADR-0001 D2 before proceeding"
        )
    return model, feature_extractor


def embed_batch(audio_batch: list[np.ndarray], sample_rate: int) -> np.ndarray:
    """Frozen pooled embeddings for a batch of equal-length 16kHz waveforms."""
    if sample_rate != ENCODER_SAMPLE_RATE:
        raise EncoderInputError(
            f"audio must already be at {ENCODER_SAMPLE_RATE}Hz "
            f"(use dsp.encoder_input.prepare_encoder_input), got {sample_rate}Hz"
        )
    model, feature_extractor = load_encoder()
    inputs = feature_extractor(
        [a.astype(np.float32) for a in audio_batch], sampling_rate=sample_rate, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.numpy()


def checkpoint_sha256() -> str:
    """sha256 of the cached model.safetensors file, for provenance (ADR-0001 D7)."""
    from huggingface_hub import hf_hub_download

    from respiralab.data.inventory import sha256_of_file

    weights_path = hf_hub_download(repo_id=CHECKPOINT, filename="model.safetensors")
    return sha256_of_file(Path(weights_path))
