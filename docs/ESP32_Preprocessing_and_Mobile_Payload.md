# ESP32 Preprocessing & Mobile Payload Format

## Overview

The system has been restructured for deployment with **hardware-on-edge preprocessing**. The ESP32 handles all signal processing, sending pre-computed features to the Raspberry Pi for model inference.

## Data Flow

```
Physical Sensor (SCT013)
    ↓
Metering IC (8 kHz sampling)
    ↓
ESP32 Preprocessing (C/C++)
    - Cycle alignment (FITPS)
    - 10-cycle windowing
    - Resampling to 400 samples
    - Normalization
    - FFT computation (200 harmonics)
    - Fryze decomposition (active/reactive)
    - ICA transformation (16 components)
    ↓
WiFi Transmission → Raspberry Pi
    ↓
Fusion-ResNet Model Inference
    ↓
Anomaly Detection & Mobile Payload
    ↓
Cloud API / Mobile Client
```

## ESP32 Preprocessed Feature Format

The ESP32 sends preprocessed features as a **JSON payload** or **NPZ binary file**:

### JSON Format (Real-time via WiFi)
```json
{
  "timestamp": 1712424225,
  "window_id": 42,
  "features": {
    "raw_window": [...400 float32...],
    "fft_magnitude": [...200 float32...],
    "fryze_active": [...50 float32...],
    "fryze_reactive": [...50 float32...],
    "ica_features": [...16 float32...]
  },
  "metadata": {
    "sample_rate": 8000,
    "cycles": 10,
    "normalizer_max": 4.523
  }
}
```

### NPZ Format (Batch Processing / Storage)
```python
# Save features (on ESP32 or Raspberry Pi):
np.savez('preprocessed_features.npz',
    raw_window=windows_raw,        # (N, 400) float32
    fft_magnitude=windows_fft,     # (N, 200) float32
    fryze_active=windows_fryze_a,  # (N, 50) float32
    fryze_reactive=windows_fryze_r,# (N, 50) float32
    ica_features=windows_ica,      # (N, 16) float32
    timestamps=timestamps_unix,    # (N,) int64 (optional)
    window_ids=window_ids          # (N,) int32 (optional)
)
```

## Feature Descriptions

| Feature | Shape | Description |
|---------|-------|-------------|
| **raw_window** | (N, 400) | Normalized current waveform (10 electrical cycles @ 50Hz) |
| **fft_magnitude** | (N, 200) | FFT magnitude spectrum (harmonics up to ~4 kHz) |
| **fryze_active** | (N, 50) | Active (real) power component over 10 cycles |
| **fryze_reactive** | (N, 50) | Reactive (imaginary) power component |
| **ica_features** | (N, 16) | Independent Component Analysis (16 components) |

All features are **float32** normalized to unit magnitude (or ICA/FFT-specific scale).

## Inference Pipeline Usage

### Option 1: Real-Time Server (REST API)
```bash
# Start the inference server
python inference_server.py --checkpoint checkpoints/best.pt \
    --device cpu --enable-anomaly-detection --port 5000

# ESP32 sends JSON POST request:
curl -X POST http://raspberrypi.local:5000/predict \
    -H "Content-Type: application/json" \
    -d @preprocessed_window.json
```

### Option 2: Batch Processing (NPZ Files)
```bash
# Process saved NPZ file
python inference_pipeline.py --checkpoint checkpoints/best.pt \
    --input preprocessed_features.npz \
    --preprocessed \
    --enable-anomaly-detection \
    --measured-current 5.2 \
    --device cpu --fp32
```

### Option 3: Directory Batch Processing
```bash
# Process all .npz files in a directory
python inference_pipeline.py --checkpoint checkpoints/best.pt \
    --input /home/pi/features_folder \
    --preprocessed \
    --enable-anomaly-detection \
    --device cpu
```

## Mobile Client Payload Format

**File: `inference_results/mobile_payload.json`**

```json
{
  "metadata": {
    "timestamp": "2026-04-06T14:23:45.123Z",
    "model_version": "1.0",
    "variant": "preprocessed",
    "n_windows": 42,
    "inference_time_ms": 245.3
  },
  "summary": {
    "most_active_appliance": "Fridge",
    "avg_active_per_window": 2.3,
    "total_anomalies": 2,
    "anomaly_types": {
      "unknown_load": 1,
      "unaccounted_current": 1
    }
  },
  "appliances": {
    "Fridge": {
      "detected": true,
      "detection_rate": 95.2,
      "detections": 40,
      "avg_confidence": 0.91,
      "max_confidence": 0.98,
      "min_confidence": 0.73
    },
    "Microwave": {
      "detected": false,
      "detection_rate": 0.0,
      "detections": 0,
      "avg_confidence": 0.0,
      "max_confidence": 0.12,
      "min_confidence": 0.01
    }
    ...
  },
  "anomalies": [
    {
      "window": 15,
      "type": "unknown_load",
      "severity": "medium",
      "message": "No appliance >50% confidence",
      "timestamp": "2026-04-06T14:25:30Z"
    }
  ],
  "recent_windows": [
    {
      "window": 38,
      "active_appliances": [
        {"name": "Fridge", "confidence": 0.94},
        {"name": "Fan", "confidence": 0.68}
      ],
      "n_active": 2
    }
    ...
  ]
}
```

## Integration with Mobile App

The mobile app/dashboard should:
1. **Poll** `/inference_results/mobile_payload.json` every N seconds
2. **Parse** the payload and update UI with:
   - Real-time appliance load detection
   - Confidence scores for each appliance
   - Anomaly alerts (unknown loads, degradation, current mismatches)
   - Historical trends (avg_confidence over time)
3. **Display**:
   - Timeline view of appliance activity
   - Energy breakdown by appliance
   - Fault/anomaly dashboard
   - 7-day health trends

## Model Input Changes

### Old (Raw Data)
```
Raw Signal (30 kHz) → Preprocessing Pipeline → Model
- Signal resampling & normalization
- ICA decomposition (computed in model)
- FFT computation (computed in model)
- Fryze decomposition (computed in model)
```

### New (Preprocessed)
```
ESP32 Preprocessing → Pre-processed Features → Model
- Raw signal (normalized)
- ICA features (pre-computed)
- FFT magnitude (pre-computed)
- Fryze components (pre-computed)
- ↓
- Model runs Conv ResNet branches + Fusion + Classifier
- ↓
- Predictions → Mobile Payload
```

**Key Change**: The model now receives **pre-computed features** instead of raw signals. The preprocessing layers in the model are bypassed using `run_inference_preprocessed()`.

## Backward Compatibility

The inference pipeline still supports raw signal input:
```bash
# Legacy: raw CSV/NPY input (preprocessing done on Pi)
python inference_pipeline.py --checkpoint best.pt \
    --input recording.csv --sample-rate 30000
```

But for production deployment with ESP32, use `--preprocessed` mode.

## Model Variant Support

Both `--variant full` and `--variant lite` are supported:
- **full**: 1.3M parameters, best for cloud
- **lite**: 250K parameters, optimized for Raspberry Pi (~50ms inference)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, pandas, scikit-learn
- Flask (optional, for real-time server)

## Examples

### Generate Example NPZ File
```python
import numpy as np

# Simulate N windows of preprocessed features
N = 100
np.savez('example_features.npz',
    raw_window=np.random.randn(N, 400).astype(np.float32),
    fft_magnitude=np.random.rand(N, 200).astype(np.float32),
    fryze_active=np.random.randn(N, 50).astype(np.float32),
    fryze_reactive=np.random.randn(N, 50).astype(np.float32),
    ica_features=np.random.randn(N, 16).astype(np.float32),
    timestamps=np.arange(N, dtype=np.int64)
)
```

### Test Inference Server
```bash
# Terminal 1: Start server
python inference_server.py --checkpoint checkpoints/best.pt --port 5000

# Terminal 2: Send a test request
python
import requests, json, numpy as np
features = {
    'features': {
        'raw_window': np.random.randn(400).tolist(),
        'fft_magnitude': np.random.rand(200).tolist(),
        'fryze_active': np.random.randn(50).tolist(),
        'fryze_reactive': np.random.randn(50).tolist(),
        'ica_features': np.random.randn(16).tolist(),
    },
    'measured_current': 5.2
}
r = requests.post('http://localhost:5000/predict', json=features)
print(json.dumps(r.json(), indent=2))
```
