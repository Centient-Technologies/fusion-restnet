# Fusion-ResNet for NILM Energy Disaggregation

A multi-branch deep learning architecture for Non-Intrusive Load Monitoring (NILM) that fuses multiple feature representations through convolutional ResNet blocks and attention-based fusion for multi-label appliance classification.

## Architecture

```
Input Signal (batch, 400)
    ├── Branch 1: Raw Signal → 1D Conv ResNet + SE Attention
    ├── Branch 2: ICA Decomposition → 1D Conv ResNet + SE Attention
    ├── Branch 3: Fryze Power Decomposition → 1D Conv ResNet + SE Attention
    └── Branch 4: FFT Frequency Domain → 1D Conv ResNet + SE Attention
                        │
                Attention-Weighted Fusion
                        │
                Classification Head → (batch, n_classes)
```

## Dataset

Uses the [PLAID](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619) (Plug Load Appliance Identification Dataset) with 16 appliance classes:

Air Conditioner, Blender, Coffee Maker, CFL, Fan, Fridge, Hair Iron, Hairdryer, Heater, Incandescent Bulb, Laptop, Microwave, Soldering Iron, Vacuum, Washing Machine, Water Kettle

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train full model (Google Colab / high-end GPU recommended)
python train_fusion_resnet.py --device cuda --variant full --epochs 250

# Train lite model (RTX 2050 / 4GB VRAM)
python train_fusion_resnet.py --device cuda --variant lite --batch-size 64 --fp32

# CPU fallback
python train_fusion_resnet.py --device cpu --variant lite --epochs 50
```

## Model Variants

| Variant | Parameters | VRAM (fp32) | Use Case |
|---------|-----------|-------------|----------|
| `full`  | ~991K     | ~1.5 GB     | Colab / RTX 3060+ |
| `lite`  | ~251K     | ~0.5 GB     | RTX 2050 / laptop GPU |

## Evaluation Output

The training script automatically generates:

- **Comprehensive metrics**: F1 (samples/macro/micro/weighted), Precision, Recall, Exact Match Accuracy, Hamming Loss, Jaccard Score
- **Per-appliance breakdown**: F1, Precision, Recall for each of the 16 appliance classes
- **Per-complexity analysis**: Performance vs number of simultaneously active appliances

### Generated Plots (`figures/`)

| Plot | Description |
|------|-------------|
| `training_curves.png` | Train/Val loss and F1 over epochs |
| `per_appliance_f1.png` | Per-class F1, Precision, Recall bar chart |
| `f1_by_components.png` | F1 vs mixture complexity (# active appliances) |
| `metrics_heatmap.png` | Per-class metrics heatmap |
| `lr_schedule.png` | Learning rate schedule |
| `dashboard.png` | Combined summary dashboard |
| `test_metrics.json` | All metrics in machine-readable format |

## Project Structure

```
├── fusion_resnet.py          # Model architecture
├── train_fusion_resnet.py    # Training, evaluation, plotting
├── fryze_utils.py            # Fryze power decomposition utilities
├── data_preprocessing.py     # Raw PLAID data processing (optional)
├── requirements.txt          # Dependencies
├── data/                     # Pre-processed .npy datasets
│   ├── X_real.npy
│   ├── X_synth.npy
│   ├── y_real.npy
│   ├── y_synth.npy
│   └── real_label_encoder.npy
├── figures/                  # Generated plots & metrics
└── checkpoints/              # Saved model weights
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- See `requirements.txt` for full list

## References

- Original ICAResNetFFN project: [ML2023SK Team 37](https://github.com/arx7ti/ML2023SK-final-project)
- PLAID Dataset: [Figshare](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619)



