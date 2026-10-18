from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from respiralab.dsp.spectrogram import compute_log_mel
from respiralab.inference.predict import InputError, analyze_recording, build_frozen_predictor

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FEATURE_MANIFEST = "manifests/feature_manifest.csv"
FEATURE_CACHE_DIR = "artifacts/features_cache"

# Two instrument states, not a palette swap: light = strip-chart paper
# (ECG/spirometer recorder), scope = phosphor-green CRT. See .impeccable.md.
THEMES = {
    "light": {
        "paper": "#F3ECE6",
        "ink": "#1F2A2E",
        "grid": "#D8A99A",
        "trace": "#1B6B63",
        "amber": "#C8762A",
        "caution": "#A23B2E",
    },
    "scope": {
        "paper": "#0E1412",
        "ink": "#E8E2D5",
        "grid": "#24312C",
        "trace": "#8FBF6B",
        "amber": "#D99A4E",
        "caution": "#C9604A",
    },
}


@st.cache_resource
def get_predictor():
    return build_frozen_predictor(FEATURE_MANIFEST, FEATURE_CACHE_DIR)


def inject_css(tokens: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=Martian+Mono:wght@400;600&display=swap');

        .stApp {{
            background-color: {tokens["paper"]};
            background-image:
                repeating-linear-gradient(0deg, {tokens["grid"]}22 0px, {tokens["grid"]}22 1px, transparent 1px, transparent 24px),
                repeating-linear-gradient(90deg, {tokens["grid"]}22 0px, {tokens["grid"]}22 1px, transparent 1px, transparent 24px);
            color: {tokens["ink"]};
        }}
        h1, h2, h3, p, label, span, div {{ color: {tokens["ink"]}; }}
        h1 {{ font-family: 'Fraunces', serif; letter-spacing: 0.01em; }}
        .lab-rule {{ border: none; border-top: 1px solid {tokens["grid"]}; margin: 1.1rem 0; }}
        .lab-eyebrow {{
            font-family: 'Martian Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.75;
            margin: 0.9rem 0 0.4rem 0;
        }}
        .lab-caution {{
            border: 1px solid {tokens["caution"]};
            border-left: 6px solid {tokens["caution"]};
            background: {tokens["caution"]}14;
            padding: 0.6rem 0.9rem;
            font-family: 'Martian Mono', monospace;
            font-size: 0.82rem;
            line-height: 1.5;
            margin: 0.6rem 0 1.2rem 0;
        }}
        .strip-trace {{
            width: 100%;
            border: 1px solid {tokens["grid"]};
            animation: pen-draw 1.1s cubic-bezier(0.16, 1, 0.3, 1) both;
        }}
        @keyframes pen-draw {{
            from {{ clip-path: inset(0 100% 0 0); }}
            to {{ clip-path: inset(0 0 0 0); }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .strip-trace {{ animation: none; }}
        }}
        .gauge-row {{
            display: grid;
            grid-template-columns: 6rem 1fr 4rem;
            align-items: center;
            gap: 0.6rem;
            margin: 0.35rem 0;
            font-family: 'Martian Mono', monospace;
            font-size: 0.85rem;
        }}
        .gauge-label {{ text-transform: uppercase; letter-spacing: 0.04em; }}
        .gauge-track {{ background: {tokens["grid"]}33; border: 1px solid {tokens["grid"]}; height: 0.9rem; }}
        .gauge-fill {{ background: {tokens["trace"]}; height: 100%; }}
        .gauge-value {{ text-align: right; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_strip_chart(audio: np.ndarray, sample_rate: int, tokens: dict[str, str]) -> str:
    log_mel = compute_log_mel(audio, sample_rate)
    cmap = LinearSegmentedColormap.from_list(
        "strip", [tokens["paper"], tokens["trace"], tokens["amber"]]
    )
    t = np.arange(len(audio)) / sample_rate
    t_end = t[-1] if t[-1] > t[0] else t[0] + 1

    fig, (wave_ax, spec_ax) = plt.subplots(
        2, 1, figsize=(9, 4.2), gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.08}
    )
    fig.patch.set_facecolor(tokens["paper"])

    wave_ax.set_facecolor(tokens["paper"])
    wave_ax.plot(t, audio, color=tokens["trace"], linewidth=0.8)
    wave_ax.set_xlim(t[0], t_end)
    wave_ax.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    wave_ax.text(
        0.0, 1.05, "WAVEFORM", transform=wave_ax.transAxes,
        fontsize=8, color=tokens["ink"], family="monospace", va="bottom",
    )

    spec_ax.set_facecolor(tokens["paper"])
    spec_ax.imshow(
        log_mel, aspect="auto", origin="lower", cmap=cmap,
        extent=(t[0], t_end, 0, log_mel.shape[0]),
    )
    spec_ax.tick_params(colors=tokens["ink"], labelsize=7, left=False, labelleft=False)
    spec_ax.set_xlabel("time (s)", fontsize=8, color=tokens["ink"], family="monospace")
    spec_ax.text(
        0.0, 1.03, "LOG-MEL SPECTROGRAM (dB)", transform=spec_ax.transAxes,
        fontsize=8, color=tokens["ink"], family="monospace", va="bottom",
    )

    for ax in (wave_ax, spec_ax):
        for spine in ax.spines.values():
            spine.set_color(tokens["grid"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=tokens["paper"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_gauges(label_names: list[str], probabilities: list[float]) -> str:
    rows = []
    for name, prob in zip(label_names, probabilities, strict=True):
        pct = max(0.0, min(100.0, prob * 100))
        rows.append(
            '<div class="gauge-row">'
            f'<span class="gauge-label">{name}</span>'
            f'<div class="gauge-track"><div class="gauge-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="gauge-value">{pct:.1f}%</span>'
            "</div>"
        )
    return "".join(rows)


def main() -> None:
    st.set_page_config(page_title="RespiraLab", page_icon="\U0001fac1", layout="centered")

    scope_mode = st.session_state.get("scope_mode", False)
    tokens = THEMES["scope"] if scope_mode else THEMES["light"]
    inject_css(tokens)

    header_col, toggle_col = st.columns([4, 1])
    with header_col:
        st.markdown("<div class='lab-eyebrow'>respiratory sound explorer</div>", unsafe_allow_html=True)
        st.title("RespiraLab")
    with toggle_col:
        st.toggle("Scope", key="scope_mode", help="Dark instrument-panel theme")

    st.markdown(
        "<div class='lab-caution'>&#9650; <strong>Experimental research tool, not a diagnostic "
        "device.</strong> Trained on the ICBHI 2017 dataset's own recording devices (stethoscope "
        "microphones) — results on a phone/laptop microphone are a domain shift and may not be "
        "reliable. A sound label is not a disease diagnosis.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='lab-eyebrow'>input</div>", unsafe_allow_html=True)
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

    audio = np.asarray(audio)

    st.markdown("<div class='lab-eyebrow'>recording strip</div>", unsafe_allow_html=True)
    strip_b64 = render_strip_chart(audio, sample_rate, tokens)
    st.markdown(f'<img class="strip-trace" src="data:image/png;base64,{strip_b64}" />', unsafe_allow_html=True)

    st.markdown("<hr class='lab-rule' />", unsafe_allow_html=True)
    st.markdown("<div class='lab-eyebrow'>cycle bounds (optional)</div>", unsafe_allow_html=True)
    use_bounds = st.checkbox("Specify expert cycle bounds (annotated_cycle mode)")
    bounds = None
    if use_bounds:
        duration_s = len(audio) / sample_rate
        start_s = st.number_input("start (s)", min_value=0.0, max_value=duration_s, value=0.0)
        end_s = st.number_input("end (s)", min_value=0.0, max_value=duration_s, value=duration_s)
        bounds = (start_s, end_s)

    try:
        predictor = get_predictor()
        result = analyze_recording(predictor, audio, sample_rate, bounds=bounds)
    except InputError as exc:
        st.error(f"Could not analyze this audio: {exc}")
        return

    st.markdown("<hr class='lab-rule' />", unsafe_allow_html=True)
    mode_label = (
        "expert-annotated cycle"
        if result["analysis_mode"] == "annotated_cycle"
        else "exploratory window — not benchmark-equivalent"
    )
    st.markdown(
        f"<div class='lab-eyebrow'>readout &middot; mode: {mode_label} &middot; "
        f"model: {result['model_id']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(render_gauges(result["label_names"], result["probabilities"]), unsafe_allow_html=True)
    st.markdown(f'<div class="lab-caution">&#9432; {result["warning"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
