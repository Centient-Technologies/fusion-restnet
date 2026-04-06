# Fusion-ResNet for NILM

## What This Project Does

This project identifies which household appliances are turned on just by looking at the total electrical current flowing into a house. This is called **Non-Intrusive Load Monitoring (NILM)** — "non-intrusive" because you only need one sensor at the main power line, not a separate sensor on every appliance.

Every appliance draws current in a unique pattern. When multiple appliances run at the same time, their patterns overlap. This model learns to untangle those overlapping patterns and identify which appliances are active.

## System Architecture

The system is designed for deployment with preprocessing on edge hardware:

```
┌─ HARDWARE LAYER (ESP32 + Metering IC) ────────┐
│                                               │
│  SCT013 Sensor → Metering IC (8 kHz sampling)│
│      ↓                                        │
│  ┌─ ESP32 Preprocessing ─────────────────┐   │
│  │ • Cycle alignment                     │   │
│  │ • Windowing (10 cycles)               │   │
│  │ • Resampling to 400 samples           │   │
│  │ • Normalization                       │   │
│  │ • FFT (200 harmonics)                 │   │
│  │ • Fryze decomposition (active/reactive)   │
│  │ • ICA transformation (16 components) │   │
│  └───────────────────────────────────────┘   │
└───────────────┬────────────────────────────────┘
                │ WiFi (pre-processed features)
                ▼
┌─ INFERENCE LAYER (Raspberry Pi / Edge Server) ┐
│                                               │
│  Fusion-ResNet Model:                        │
│  • Branch 1: Raw signal → Conv ResNet        │
│  • Branch 2: ICA features → Conv ResNet      │
│  • Branch 3: Fryze (active/reactive)         │
│  • Branch 4: FFT (frequency content)         │
│  • Attention-weighted fusion                 │
│  • Multi-label classifier                    │
│                                               │
│  Anomaly Detection:                          │
│  • Unknown load detection                    │
│  • Appliance degradation tracking            │
│  • Unaccounted current validation            │
│                                               │
│  Output: {appliances, confidence, anomalies} │
└───────────────┬────────────────────────────────┘
                │ WiFi / REST API
                ▼
┌─ CLIENT LAYER (Mobile App) ────────────────────┐
│                                               │
│  • Real-time appliance status & confidence  │
│  • Anomaly alerts                           │
│  • Energy consumption breakdown              │
│  • Historical trends & usage patterns       │
│                                               │
└───────────────────────────────────────────────┘
```

## What the Model Expects (Input)

The Fusion-ResNet model is trained to accept **pre-processed features** from the ESP32:

| Feature | Shape | Description |
|---------|-------|-------------|
| **raw_window** | (400,) | Normalized current waveform (10 cycles) |
| **fft_magnitude** | (200,) | FFT magnitude spectrum (harmonics) |
| **fryze_active** | (50,) | Active current (useful power) |
| **fryze_reactive** | (50,) | Reactive current (wasted power) |
| **ica_features** | (16,) | ICA-decomposed components |

All preprocessing happens on the ESP32 — the model receives feature vectors, not raw signals.

## What the Model Outputs

For each inference window, the model outputs:

1. **Predictions**: Binary classification for each appliance (ON/OFF)
2. **Confidence Scores**: Probability [0, 1] for each appliance
3. **Anomalies**: List of detected faults or unusual patterns
4. **Appliance Health**: Trending confidence and degradation warnings

**Mobile Payload Example:**
```json
{
  "timestamp": "2026-04-06T14:23:45Z",
  "window_id": 42,
  "predictions": {
    "Fridge": {"active": true, "confidence": 0.94},
    "Microwave": {"active": true, "confidence": 0.87},
    "Laptop": {"active": false, "confidence": 0.12}
  },
  "anomalies": [
    {
      "type": "unknown_load",
      "severity": "medium",
      "message": "Unaccounted current: 6.5A measured, 5.2A predicted"
    }
  ],
  "health": {
    "Fridge": {
      "status": "healthy",
      "avg_confidence": 0.91,
      "trend": "stable"
    }
  },
  "metadata": {
    "model_version": "1.0",
    "inference_time_ms": 12,
    "anomaly_detection_enabled": true
  }
}
```

The model can detect these **15 appliances**:
> Air Conditioner, Blender, Coffee Maker, Compact Fluorescent Lamp, Fan, Fridge, Hair Iron, Hairdryer, Heater, Incandescent Light Bulb, Laptop, Microwave, Soldering Iron, Vacuum, Washing Machine

## Fault Detection & Anomaly Monitoring

The system includes real-time anomaly detection for electrical faults:

| Anomaly Type | Trigger | Use Case |
|---|---|---|
| **Unknown Load** | No appliance >50% confidence | Unaccounted consumption |
| **Degradation** | Confidence drops >15% over 7 days | Appliance wear prediction |
| **Unaccounted Current** | Measured > predicted by >20% | Ground faults, leakage |

Anomalies are tracked over time and can trigger preventive maintenance alerts.

## Dataset

Trained on [PLAID](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619) — real electrical measurements from US households at 30 kHz.

During training, individual appliance recordings are mixed to simulate real household scenarios where multiple appliances run simultaneously.

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Train (One-Time, on Colab or GPU Machine)

```bash
# Google Colab with T4 GPU (recommended)
python train_fusion_resnet.py --device cuda --variant full --epochs 250 --fp32

# Laptop GPU (RTX 2050 / 4GB VRAM)
python train_fusion_resnet.py --device cuda --variant lite --batch-size 64 --fp32

# CPU only
python train_fusion_resnet.py --device cpu --variant lite --epochs 50
```

### Run Inference (Deployment, on Raspberry Pi)

The inference pipeline is designed to accept pre-processed features from the ESP32:

```bash
# Receive real-time features from ESP32 via REST API
python inference_server.py --checkpoint checkpoints/fusion_resnet/best.pt \
    --device cpu --enable-anomaly-detection --port 5000

# Or batch inference on stored pre-processed data
python inference_pipeline.py --checkpoint checkpoints/fusion_resnet/best.pt \
    --input preprocessed_features.npz --device cpu --fp32 \
    --enable-anomaly-detection --measured-current 5.2
```

Results are saved to `inference_results/` with:
- `predictions.json` — Full results with anomalies
- `predictions.csv` — Tabular summary
- `.anomaly_history.json` — Appliance health tracking

### Hardware Preprocessing (ESP32)

The ESP32 handles all data preprocessing before sending to the inference server. See [Hardware_and_Preprocessing_Walkthrough.md](docs/Hardware_and_Preprocessing_Walkthrough.md) for implementation details.

**What the ESP32 Sends:**
```json
{
  "timestamp": 1712424225,
  "features": {
    "raw_window": [...400 floats...],
    "fft_magnitude": [...200 floats...],
    "fryze_active": [...50 floats...],
    "fryze_reactive": [...50 floats...],
    "ica_features": [...16 floats...]
  }
}
```

## Model Variants

| Variant | Parameters | VRAM | Best For |
|---------|-----------|------|----------|
| `full`  | ~1.3M    | ~1.5 GB (fp32) | Cloud / server deployment |
| `lite`  | ~250K    | ~0.5 GB (fp32) | Raspberry Pi / edge device |

## Training Output

The training script generates:

- **Checkpoints** in `checkpoints/fusion_resnet/` — best and last model weights
- **Metrics** in `figures/test_metrics.json` — comprehensive evaluation
- **Plots**:
  - `training_curves.png` — Loss and F1 over epochs
  - `per_appliance_f1.png` — Per-appliance performance
  - `f1_by_components.png` — Performance vs simultaneous appliances
  - `dashboard.png` — Executive summary

## Project Structure

```
├── fusion_resnet.py              # Model architecture
├── train_fusion_resnet.py        # Training pipeline
├── inference_pipeline.py         # Batch inference on pre-processed features
├── inference_server.py           # Real-time REST API for ESP32 (optional)
├── anomaly_detector.py           # Fault & degradation detection
├── fryze_utils.py                # Power decomposition utilities
├── data_preprocessing.py         # PLAID raw data processing
├── docs/
│   └── Hardware_and_Preprocessing_Walkthrough.md
├── checkpoints/                  # Trained model weights
├── figures/                      # Generated metrics & plots
└── data/                         # Training datasets
```

## Deployment Options

### Option 1: Edge-Only (Recommended for Privacy)
- ESP32: preprocessing only
- Raspberry Pi: full inference + anomaly detection
- Cloud: results storage & mobile API only
- **Advantage**: Raw data never leaves the house

### Option 2: Cloud Inference
- ESP32: preprocessing
- Cloud: model inference + anomaly detection
- **Advantage**: No local compute needed, pay per inference

### Option 3: Hybrid
- ESP32: preprocessing
- Local Pi: real-time alerts
- Cloud: long-term storage + app backend
- **Advantage**: Best of both worlds

## Requirements

- Python 3.10+
- PyTorch 2.0+
- ESP32 with metering IC (ADE7953, ATM90E26, etc.) for real deployment
- Optional: Raspberry Pi 4+ for local inference (~$35)

## References

- Original ICAResNetFFN project: [ML2023SK Team 37](https://github.com/arx7ti/ML2023SK-final-project)
- PLAID Dataset: [Figshare](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619)




