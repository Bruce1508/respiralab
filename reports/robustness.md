# D31 - Fixed-SNR robustness check (no model changes)

Frozen E01 pipeline (identical fit as D29/D30), re-evaluated on the SAME
primary test cycles with additive white Gaussian noise injected at fixed
SNR levels before feature extraction. Model itself is untouched.

**Noise method:** additive white Gaussian noise, power scaled per-cycle to
hit the target SNR (`noise_power = signal_power / 10**(SNR_dB/10)`), applied
to the resampled 4000Hz cycle waveform before handcrafted feature extraction.
**SNR values tested:** [20, 10, 0] dB.

## Results vs clean baseline (D29, macro_f1=0.2861)

| SNR (dB) | macro_f1 | icbhi_score | delta vs clean |
|---|---:|---:|---:|
| 20 | 0.2384 | 0.4314 | -0.0477 |
| 10 | 0.2533 | 0.4834 | -0.0328 |
| 0 | 0.1927 | 0.4924 | -0.0934 |
| clean (D29) | 0.2861 | 0.3762 | 0.0000 |

## Per-class detail

### SNR=20dB
```json
{
  "normal": {
    "precision": 0.6374367622259697,
    "recall": 0.5206611570247934,
    "f1": 0.5731614859742229,
    "support": 726
  },
  "crackle": {
    "precision": 0.2596685082872928,
    "recall": 0.597457627118644,
    "f1": 0.362002567394095,
    "support": 236
  },
  "wheeze": {
    "precision": 0.5,
    "recall": 0.009433962264150943,
    "f1": 0.018518518518518517,
    "support": 106
  },
  "both": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "support": 73
  }
}
```
### SNR=10dB
```json
{
  "normal": {
    "precision": 0.6392896781354052,
    "recall": 0.7933884297520661,
    "f1": 0.7080516287645974,
    "support": 726
  },
  "crackle": {
    "precision": 0.3050847457627119,
    "recall": 0.3050847457627119,
    "f1": 0.3050847457627119,
    "support": 236
  },
  "wheeze": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "support": 106
  },
  "both": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "support": 73
  }
}
```
### SNR=0dB
```json
{
  "normal": {
    "precision": 0.633303808680248,
    "recall": 0.9848484848484849,
    "f1": 0.77088948787062,
    "support": 726
  },
  "crackle": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "support": 236
  },
  "wheeze": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "support": 106
  },
  "both": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "support": 73
  }
}
```

## Interpretation

Macro-F1 degrades at every tested noise level vs the clean D29 baseline
(worst at 0dB SNR: -0.0934), while ICBHI Score rises with noise - the same
Normal-specificity blind spot flagged in D21/D27 (noise pushes more cycles
toward a "normal-looking" prediction, inflating specificity while abnormal
sensitivity suffers). Per `planning/DECISION_RULES.md` ("robustness
collapses under noise... narrow supported use claim, don't assert
generalization"): this model's reported performance applies to recordings
at a noise level comparable to the ICBHI dataset's own acquisition
conditions, not to arbitrary noisy consumer-microphone input.
