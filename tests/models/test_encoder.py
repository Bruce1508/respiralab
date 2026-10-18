from __future__ import annotations

import numpy as np
import pytest

from respiralab.dsp.encoder_input import ENCODER_SAMPLE_RATE
from respiralab.models.encoder import (
    CHECKPOINT_SAMPLE_RATE,
    EMBEDDING_DIM,
    EncoderInputError,
    checkpoint_sha256,
    embed_batch,
    load_encoder,
)


def test_embed_batch_rejects_wrong_sample_rate():
    audio = np.zeros(1000, dtype=np.float32)
    with pytest.raises(EncoderInputError, match="16000"):
        embed_batch([audio], sample_rate=4000)


def _encoder_available() -> bool:
    try:
        load_encoder()
    except Exception:  # noqa: BLE001 -- any failure (no network, HF outage, bad cache) means "skip"
        return False
    return True


@pytest.mark.skipif(
    not _encoder_available(),
    reason="requires downloading the AST checkpoint from Hugging Face Hub (network, ~330MB)",
)
def test_checkpoint_declares_the_expected_sample_rate():
    assert CHECKPOINT_SAMPLE_RATE == ENCODER_SAMPLE_RATE == 16000


@pytest.mark.skipif(
    not _encoder_available(),
    reason="requires downloading the AST checkpoint from Hugging Face Hub (network, ~330MB)",
)
def test_embed_batch_is_deterministic_for_the_same_input():
    rng = np.random.default_rng(0)
    audio = rng.normal(size=ENCODER_SAMPLE_RATE * 1).astype(np.float32)

    emb1 = embed_batch([audio], sample_rate=ENCODER_SAMPLE_RATE)
    emb2 = embed_batch([audio], sample_rate=ENCODER_SAMPLE_RATE)

    assert emb1.shape == (1, EMBEDDING_DIM)
    np.testing.assert_array_equal(emb1, emb2)


@pytest.mark.skipif(
    not _encoder_available(),
    reason="requires downloading the AST checkpoint from Hugging Face Hub (network, ~330MB)",
)
def test_embed_batch_gives_different_output_for_different_input():
    rng = np.random.default_rng(1)
    audio_a = rng.normal(size=ENCODER_SAMPLE_RATE).astype(np.float32)
    audio_b = rng.normal(size=ENCODER_SAMPLE_RATE).astype(np.float32) * 5 + 2

    emb_a = embed_batch([audio_a], sample_rate=ENCODER_SAMPLE_RATE)
    emb_b = embed_batch([audio_b], sample_rate=ENCODER_SAMPLE_RATE)

    assert not np.allclose(emb_a, emb_b)


@pytest.mark.skipif(
    not _encoder_available(),
    reason="requires downloading the AST checkpoint from Hugging Face Hub (network, ~330MB)",
)
def test_checkpoint_sha256_is_a_stable_64char_hex_digest():
    h1 = checkpoint_sha256()
    h2 = checkpoint_sha256()
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # raises if not valid hex
