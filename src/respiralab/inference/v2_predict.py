"""D56: v2 (AST fine-tune) serving -- hash-verified weights + OOD gate (ADR-0001 D7).

v2's weights are too large for git (328.86MB > GitHub's 100MB limit, forced
the D55 history rewrite) -- they live on Hugging Face Hub as a versioned
external artifact, downloaded and sha256-verified against
`results/v2_frozen/v2_freeze.json` before any model is constructed. A hash
mismatch is a hard refusal, never a silent fallback to possibly-wrong
weights. An out-of-distribution input is likewise refused, not predicted on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import ASTFeatureExtractor

from respiralab.data.inventory import sha256_of_file
from respiralab.dsp.encoder_input import ENCODER_SAMPLE_RATE, prepare_encoder_input
from respiralab.evaluation.calibration import softmax
from respiralab.evaluation.metrics import LABEL_ORDER
from respiralab.inference.predict import NOT_DIAGNOSTIC_WARNING, InputError, validate_wav
from respiralab.models.encoder import CHECKPOINT
from respiralab.models.finetune import ASTClassifier, select_device
from respiralab.models.ood_gate import is_in_distribution, load_gate

MODEL_ID = "v2_E06_bounded_finetune_frozen"
UNFREEZE_LAST_N = 2  # must match D52's frozen architecture


class WeightsHashMismatchError(RuntimeError):
    """Raised when downloaded weights do not match the frozen provenance hash."""


class OutOfDistributionError(InputError):
    """Raised when input is too far from the train-fit distribution to trust a prediction."""


def resolve_weights(repo_id: str, filename: str, expected_sha256: str, revision: str | None = None) -> Path:
    """Downloads from HF Hub (cached) and refuses to return a path on hash mismatch."""
    local_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    actual = sha256_of_file(local_path)
    if actual != expected_sha256:
        raise WeightsHashMismatchError(
            f"{filename} from {repo_id}@{revision or 'main'} has sha256={actual}, "
            f"expected {expected_sha256}. Refusing to load."
        )
    return local_path


def build_v2_predictor(
    repo_id: str,
    v2_freeze_json: str | Path,
    ood_gate_json: str | Path,
    revision: str | None = None,
) -> dict:
    """Downloads+verifies v2 weights, loads the model, feature extractor and OOD gate."""
    freeze = json.loads(Path(v2_freeze_json).read_text())
    gate = load_gate(json.loads(Path(ood_gate_json).read_text()))
    device = select_device()

    weights_path = resolve_weights(
        repo_id=repo_id,
        filename=Path(freeze["final_model"]["weights_path"]).name,
        expected_sha256=freeze["final_model"]["weights_sha256"],
        revision=revision,
    )
    model = ASTClassifier(unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    return {
        "model": model,
        "feature_extractor": ASTFeatureExtractor.from_pretrained(CHECKPOINT),
        "device": device,
        "temperature": freeze["calibration"]["temperature"],
        "ood_gate": gate,
    }


def analyze_cycle_v2(predictor: dict, audio: np.ndarray, sample_rate: int) -> dict:
    validate_wav(audio, sample_rate)
    waveform = prepare_encoder_input(np.asarray(audio), sample_rate).astype(np.float32)
    inputs = predictor["feature_extractor"]([waveform], sampling_rate=ENCODER_SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(predictor["device"]) for k, v in inputs.items()}

    model = predictor["model"]
    with torch.no_grad():
        pooled = model.encoder(**inputs).pooler_output
        logits = model.classifier(pooled).cpu().numpy()

    embedding = pooled.cpu().numpy()[0]
    in_distribution, distance = is_in_distribution(embedding, predictor["ood_gate"])
    if not in_distribution:
        raise OutOfDistributionError(
            f"input is {distance:.2f} Mahalanobis units from the training distribution "
            f"(train-fit threshold {predictor['ood_gate']['threshold']:.2f}); refusing to predict."
        )

    probs = softmax(logits / predictor["temperature"])[0]
    return {
        "model_id": MODEL_ID,
        "label_names": list(LABEL_ORDER),
        "probabilities": [float(p) for p in probs],
        "warning": NOT_DIAGNOSTIC_WARNING,
        "ood_distance": distance,
        "ood_threshold": predictor["ood_gate"]["threshold"],
        "analysis_mode": "annotated_cycle",
    }


def analyze_recording_v2(
    predictor: dict, audio: np.ndarray, sample_rate: int, bounds: tuple[float, float] | None = None
) -> dict:
    audio = np.asarray(audio)
    if bounds is not None:
        start_s, end_s = bounds
        start_idx, end_idx = int(start_s * sample_rate), int(end_s * sample_rate)
        if not (0 <= start_idx < end_idx <= len(audio)):
            raise InputError(f"invalid cycle bounds: start={start_s}, end={end_s}")
        segment = audio[start_idx:end_idx]
        mode = "annotated_cycle"
    else:
        segment = audio
        mode = "exploratory_window"

    result = analyze_cycle_v2(predictor, segment, sample_rate)
    result["analysis_mode"] = mode
    if mode == "exploratory_window":
        result["warning"] += (
            " Whole-recording input without expert cycle bounds is exploratory, "
            "not benchmark-equivalent."
        )
    return result
