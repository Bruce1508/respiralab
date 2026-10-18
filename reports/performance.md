# D40 - Inference latency/memory benchmark

**Tagged config:** commit `3ac1c1cd113d3f46da1d17b68a37b63b1d9a2946`, frozen model `E01_logistic_regression_frozen_cdc0b4f`
(`reports/model_freeze.md`). Benchmarked `analyze_cycle()` from `inference/predict.py`
(D35) - resample + handcrafted feature extraction + predict_proba, per call.

## Hardware
- Platform: macOS-26.6.2-arm64-arm-64bit
- Processor: arm
- Python: 3.11.15

## Input size
- n=200 real primary-test cycles (random sample, seed=0)
- cycle duration: min=0.33s, max=5.95s, mean=2.98s

## Latency (CPU, single-threaded call per cycle)
- p50: 2.43 ms
- p95: 3.54 ms
- mean: 5.84 ms

## Memory
- Peak process RSS during benchmark: 700.7 MB (includes the fitted
  sklearn pipeline + numpy/scipy/librosa import overhead, not isolated per-call)
