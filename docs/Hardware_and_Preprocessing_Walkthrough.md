# Fusion-ResNet NILM: Hardware & Preprocessing Walkthrough

## Overview

The intended runtime is now **hardware-only**:

- the device captures the aggregate current waveform
- preprocessing is done on the device
- the model can also run on the device

This repository stays focused on training, evaluation, and offline validation of that pipeline.

## Signal Path

```text
Sensor / metering IC
  -> raw current samples
  -> cycle-aligned windowing
  -> resampling to 400 samples
  -> per-window normalization
  -> FFT magnitude (200 bins)
  -> Fryze active/reactive components (50 + 50)
  -> ICA features (16)
  -> Fusion-ResNet
  -> multi-label appliance predictions
```

## Why PLAID Still Fits

PLAID is recorded at 30 kHz, but the model does not consume the full raw stream directly.
The training pipeline reduces each example to a normalized 400-sample, 10-cycle window.

That means lower-rate hardware can still work if it captures enough samples per mains cycle and reproduces the same preprocessing steps before inference.

## Training View

The current training script still starts from normalized raw windows and learns the multi-branch model with:

- raw waveform branch
- ICA branch
- Fryze branch
- FFT branch

This is fine for now because it preserves the original training setup and keeps the checkpoint compatible with offline validation.

## Hardware View

For hardware execution, you can generate the same feature set on-device and feed those features into the preprocessed inference path for validation.

Expected feature shapes:

- `raw_window`: `(400,)`
- `fft_magnitude`: `(200,)`
- `fryze_active`: `(50,)`
- `fryze_reactive`: `(50,)`
- `ica_features`: `(16,)`

## Recommended Direction

If the final shipped system will always run on precomputed hardware features, the cleanest long-term design is:

1. reproduce the preprocessing exactly on NumPy data offline
2. validate it with `inference_pipeline.py --preprocessed`
3. later add a dedicated training path for precomputed features if you want full train/inference symmetry

That avoids the mismatch where part of preprocessing exists in the model during training but gets bypassed later on the device.
