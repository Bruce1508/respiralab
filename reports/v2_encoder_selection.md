# D47 — Pretrained encoder selection

**Checkpoint:** `MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer,
AudioSet-pretrained). Selected over BEATs after checking both: both are permissive (AST
BSD-3-Clause, BEATs MIT via the `microsoft/unilm` repo root LICENSE) and both confirm 16 kHz
input, matching ADR-0001 D2's assumption without needing an amendment. AST is distributed on
Hugging Face Hub with native `transformers.ASTModel` support; BEATs ships only as a GitHub
checkpoint with no HF integration, requiring a hand-written loader for no confirmed benefit here
— picked AST as the simpler, equally-compliant path.

## Provenance (`results/E05_encoder/checkpoint_provenance.json`)

| field | value |
|---|---|
| checkpoint | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| license | BSD-3-Clause |
| sample_rate | 16000 (confirmed in `preprocessor_config.json`; matches ADR-0001) |
| embedding_dim | 768 (pooled `pooler_output`) |
| params | 86,187,264 (matches the model card's "86.6M") |
| weights sha256 | `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` |

## "~60% ICBHI Score" claim — sourced

The prior session's unverified "≈60%" figure traces to Işık et al., *Geometry-Aware Optimization
for Respiratory Sound Classification* (arXiv:2512.22564), Table 1: a plain AST-fine-tuning
baseline scores ICBHI Score **59.55%** (Sp=77.14, Se=41.97). Full detail and the critical caveat
(official 60/40 split, binary Normal-vs-Abnormal — **not** this project's 4-class patient-disjoint
protocol, not directly comparable) is in `docs/07_SOURCE_VERIFICATION.md` item 7.

## Implementation

`src/respiralab/models/encoder.py`:
- `load_encoder()` — cached `ASTModel` + `ASTFeatureExtractor`, raises if the checkpoint's declared
  sample rate ever drifts from the ADR's 16 kHz assumption (fail loud, not silent).
- `embed_batch(audio_batch, sample_rate)` — frozen (`eval()`, `no_grad()`) pooled 768-d embeddings;
  rejects any input not already at 16 kHz (use `dsp.encoder_input.prepare_encoder_input` first).
- `checkpoint_sha256()` — sha256 of the cached `model.safetensors` for provenance.

## Verification

- `uv run pytest -q` -> 195 passed (190 prior + 5 new in `tests/models/test_encoder.py`). The 4 tests
  that need the real checkpoint are `skipif`-guarded on load success — they run here (checkpoint
  cached from this session's verification) but skip cleanly on a fresh clone/CI with no network, same
  pattern as `test_predict.py`'s real-feature-cache guard.
- `uv run ruff check src tests` -> All checks passed.
- Determinism: `embed_batch` on the same input twice gives bit-identical output
  (`test_embed_batch_is_deterministic_for_the_same_input`).
- Added `transformers>=4.40` to `pyproject.toml`/`uv.lock` — the only new dependency; no custom
  model/checkpoint-loading code was written.
