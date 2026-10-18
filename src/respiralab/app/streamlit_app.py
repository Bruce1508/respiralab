from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from respiralab.dsp.spectrogram import compute_log_mel
from respiralab.inference.predict import InputError, analyze_recording, build_frozen_predictor

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FEATURE_MANIFEST = "manifests/feature_manifest.csv"
FEATURE_CACHE_DIR = "artifacts/features_cache"


@st.cache_resource
def get_predictor():
    return build_frozen_predictor(FEATURE_MANIFEST, FEATURE_CACHE_DIR)


def plot_waveform_and_spectrogram(audio: np.ndarray, sample_rate: int):
    log_mel = compute_log_mel(audio, sample_rate)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    t = np.arange(len(audio)) / sample_rate
    axes[0].plot(t, audio)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("time (s)")
    axes[1].imshow(log_mel, aspect="auto", origin="lower")
    axes[1].set_title("log-Mel spectrogram (dB)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("mel bin")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="RespiraLab", page_icon="🫁")
    st.title("RespiraLab — respiratory sound explorer")
    st.warning(
        "**Experimental research tool, not a diagnostic device.** Trained on the "
        "ICBHI 2017 dataset's own recording devices (stethoscope microphones) — "
        "results on a phone/laptop microphone are a domain shift and may not be "
        "reliable. A sound label is not a disease diagnosis.",
        icon="⚠️",
    )

    uploaded_file = st.file_uploader("Upload a mono WAV file", type=["wav"])
    if uploaded_file is None:
        return

    raw_bytes = uploaded_file.getvalue()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        st.error(f"File too large: {len(raw_bytes) / 1e6:.1f} MB exceeds the 20 MB limit.")
        return

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw_bytes))
    except Exception:  # noqa: BLE001 - untrusted upload, libsndfile raises varied exception types
        st.error("Could not decode this file as audio. Please upload a valid WAV file.")
        return

    st.pyplot(plot_waveform_and_spectrogram(np.asarray(audio), sample_rate))

    st.subheader("Cycle bounds (optional)")
    use_bounds = st.checkbox("Specify expert cycle bounds (annotated_cycle mode)")
    bounds = None
    if use_bounds:
        duration_s = len(audio) / sample_rate
        start_s = st.number_input("start (s)", min_value=0.0, max_value=duration_s, value=0.0)
        end_s = st.number_input("end (s)", min_value=0.0, max_value=duration_s, value=duration_s)
        bounds = (start_s, end_s)

    try:
        predictor = get_predictor()
        result = analyze_recording(predictor, np.asarray(audio), sample_rate, bounds=bounds)
    except InputError as exc:
        st.error(f"Could not analyze this audio: {exc}")
        return

    st.subheader("Result")
    mode_label = "expert-annotated cycle" if result["analysis_mode"] == "annotated_cycle" else "exploratory window (not benchmark-equivalent)"
    st.write(f"**Mode:** {mode_label}")
    st.write(f"**Model:** `{result['model_id']}`")
    st.bar_chart(dict(zip(result["label_names"], result["probabilities"], strict=True)))
    st.info(result["warning"], icon="ℹ️")


if __name__ == "__main__":
    main()
