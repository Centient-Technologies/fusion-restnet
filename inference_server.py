"""
Inference Server for Real-Time NILM Predictions
==============================================

REST API server that receives pre-processed features from ESP32
and returns predictions + anomalies in real-time.

The model expects features that have been preprocessed on hardware:
- raw_window (400 samples, normalized)
- fft_magnitude (200 frequency bins)
- fryze_active (50 samples)
- fryze_reactive (50 samples)
- ica_features (16 components)

Usage:
    python inference_server.py --checkpoint checkpoints/fusion_resnet/best.pt \
        --device cpu --enable-anomaly-detection --port 5000

    # Test with:
    curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d @preprocessed_features.json
"""

import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import argparse
import logging

from fusion_resnet import FusionResNet, FusionResNetLite
from anomaly_detector import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionResNetPreprocessed(nn.Module):
    """
    Fusion-ResNet variant that accepts pre-processed features instead of raw waveforms.
    
    This is used during deployment when the ESP32 handles all preprocessing.
    The model skips ICA, Fryze, FFT, and normalization layers since they're done on hardware.
    """

    def __init__(self, base_model: FusionResNet):
        super().__init__()
        self.base_model = base_model
        self.n_classes = base_model.n_classes

    def forward(
        self,
        raw_window: torch.Tensor,      # (B, 400)
        fft_magnitude: torch.Tensor,   # (B, 200)
        fryze_active: torch.Tensor,    # (B, 50)
        fryze_reactive: torch.Tensor,  # (B, 50)
        ica_features: torch.Tensor,    # (B, 16)
    ) -> torch.Tensor:
        """
        Forward pass with pre-processed features.
        
        Args:
            raw_window: Normalized raw current waveform
            fft_magnitude: FFT magnitude spectrum (pre-computed harmonics)
            fryze_active: Active current component
            fryze_reactive: Reactive current component
            ica_features: ICA-decomposed components
            
        Returns:
            logits: (B, n_classes) raw scores
        """
        feats = []

        # Branch 1: Raw signal (pre-normalized)
        raw_feat = self.base_model.raw_branch(raw_window)
        feats.append(raw_feat)

        # Branch 2: ICA (features already computed)
        # Reshape ICA features to (B, 1, 16) for Conv1d processing
        ica_expanded = ica_features.unsqueeze(1)  # (B, 1, 16)
        ica_feat = self.base_model.ica_branch.stem(ica_expanded)
        ica_feat = self.base_model.ica_branch.stages(ica_feat)
        ica_feat = self.base_model.ica_branch.pool(ica_feat).squeeze(-1)
        feats.append(ica_feat)

        # Branch 3: Fryze (features already decomposed)
        # Stack active and reactive: (B, 50) + (B, 50) -> (B, 2, 50)
        fryze_stack = torch.stack([fryze_reactive, fryze_active], dim=1)
        fryze_feat = self.base_model.fryze_branch.stem(fryze_stack)
        fryze_feat = self.base_model.fryze_branch.stages(fryze_feat)
        fryze_feat = self.base_model.fryze_branch.pool(fryze_feat).squeeze(-1)
        feats.append(fryze_feat)

        # Branch 4: FFT (pre-computed magnitude)
        fft_expanded = fft_magnitude.unsqueeze(1)  # (B, 1, 200)
        fft_feat = self.base_model.fft_branch.stem(fft_expanded)
        fft_feat = self.base_model.fft_branch.stages(fft_feat)
        fft_feat = self.base_model.fft_branch.pool(fft_feat).squeeze(-1)
        feats.append(fft_feat)

        # Fuse all branches
        fused = self.base_model.fusion(feats)

        # Classify
        return self.base_model.classifier(fused)


class NILMInferenceServer:
    """Real-time NILM inference server for ESP32 integration."""

    def __init__(
        self,
        checkpoint_path: str,
        n_classes: int = 15,
        variant: str = 'full',
        device: str = 'cpu',
        fp32: bool = True,
        enable_anomaly_detection: bool = False,
    ):
        self.device = device
        self.dtype = torch.float32 if fp32 else torch.float64

        # Load base model
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        ModelClass = FusionResNet if variant == 'full' else FusionResNetLite
        self.model = ModelClass(n_classes=n_classes, signal_length=400)

        if self.dtype == torch.float64:
            self.model = self.model.double()
        else:
            self.model = self.model.float()

        self.model = self.model.to(device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.threshold = ckpt.get('threshold', 0.5)
        self.n_classes = n_classes

        # Wrap with preprocessed variant
        self.model_preprocessed = FusionResNetPreprocessed(self.model)
        self.model_preprocessed.eval()

        logger.info(f"Model loaded, threshold: {self.threshold:.2f}")

        # Anomaly detection
        self.anomaly_detector = None
        if enable_anomaly_detection:
            appliance_names = [f'Appliance_{i}' for i in range(n_classes)]
            self.anomaly_detector = AnomalyDetector(appliance_names)
            logger.info("Anomaly detection enabled")

    @torch.no_grad()
    def predict(
        self,
        raw_window: np.ndarray,      # (400,)
        fft_magnitude: np.ndarray,   # (200,)
        fryze_active: np.ndarray,    # (50,)
        fryze_reactive: np.ndarray,  # (50,)
        ica_features: np.ndarray,    # (16,)
        measured_current: float = None,
    ) -> dict:
        """
        Run inference on pre-processed features.

        Args:
            raw_window: Normalized current waveform
            fft_magnitude: FFT magnitude spectrum
            fryze_active: Active current
            fryze_reactive: Reactive current
            ica_features: ICA components
            measured_current: Optional, for anomaly detection

        Returns:
            Dict with predictions, confidence scores, and anomalies
        """
        # Convert numpy to torch, add batch dimension
        raw = torch.tensor(raw_window, dtype=self.dtype, device=self.device).unsqueeze(0)
        fft = torch.tensor(fft_magnitude, dtype=self.dtype, device=self.device).unsqueeze(0)
        fryze_a = torch.tensor(fryze_active, dtype=self.dtype, device=self.device).unsqueeze(0)
        fryze_r = torch.tensor(fryze_reactive, dtype=self.dtype, device=self.device).unsqueeze(0)
        ica = torch.tensor(ica_features, dtype=self.dtype, device=self.device).unsqueeze(0)

        # Forward pass
        logits = self.model_preprocessed(raw, fft, fryze_a, fryze_r, ica)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs >= self.threshold).astype(int)

        # Format results
        predictions = {f'appliance_{i}': {
            'active': bool(preds[i]),
            'confidence': float(probs[i]),
        } for i in range(self.n_classes)}

        # Check for anomalies
        anomalies = []
        if self.anomaly_detector:
            detected_anomalies = self.anomaly_detector.check_window(
                predictions=preds,
                probabilities=probs,
                measured_current=measured_current,
            )
            anomalies = [a.to_dict() for a in detected_anomalies]

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'predictions': predictions,
            'anomalies': anomalies,
            'metadata': {
                'threshold': self.threshold,
                'max_confidence': float(probs.max()),
                'n_active': int(preds.sum()),
            }
        }


def create_app(config: dict) -> Flask:
    """Create Flask app for inference server."""
    app = Flask(__name__)
    
    # Initialize inference server
    server = NILMInferenceServer(
        checkpoint_path=config['checkpoint'],
        n_classes=config.get('n_classes', 15),
        variant=config.get('variant', 'full'),
        device=config.get('device', 'cpu'),
        fp32=config.get('fp32', True),
        enable_anomaly_detection=config.get('enable_anomaly_detection', False),
    )

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'ready',
            'model': f'Fusion-ResNet-{config.get("variant", "full")}',
            'n_classes': config.get('n_classes', 15),
            'threshold': server.threshold,
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """Predict endpoint. Accepts pre-processed features from ESP32."""
        try:
            data = request.get_json()

            # Extract features
            features = data.get('features', {})
            raw_window = np.array(features.get('raw_window', []), dtype=np.float32)
            fft_magnitude = np.array(features.get('fft_magnitude', []), dtype=np.float32)
            fryze_active = np.array(features.get('fryze_active', []), dtype=np.float32)
            fryze_reactive = np.array(features.get('fryze_reactive', []), dtype=np.float32)
            ica_features = np.array(features.get('ica_features', []), dtype=np.float32)
            measured_current = data.get('measured_current', None)

            # Validate shapes
            assert raw_window.shape == (400,), f"raw_window shape {raw_window.shape}, expected (400,)"
            assert fft_magnitude.shape == (200,), f"fft_magnitude shape {fft_magnitude.shape}, expected (200,)"
            assert fryze_active.shape == (50,), f"fryze_active shape {fryze_active.shape}, expected (50,)"
            assert fryze_reactive.shape == (50,), f"fryze_reactive shape {fryze_reactive.shape}, expected (50,)"
            assert ica_features.shape == (16,), f"ica_features shape {ica_features.shape}, expected (16,)"

            # Run inference
            result = server.predict(
                raw_window=raw_window,
                fft_magnitude=fft_magnitude,
                fryze_active=fryze_active,
                fryze_reactive=fryze_reactive,
                ica_features=ica_features,
                measured_current=measured_current,
            )

            return jsonify(result)

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 400

    @app.route('/health_report', methods=['GET'])
    def health_report():
        """Get appliance health report."""
        if not server.anomaly_detector:
            return jsonify({'error': 'Anomaly detection not enabled'}), 400

        health = server.anomaly_detector.get_system_health(days=7)
        return jsonify(health)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NILM Inference Server')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--fp32', action='store_true', help='Use float32')
    parser.add_argument('--variant', type=str, default='full', choices=['full', 'lite'])
    parser.add_argument('--n-classes', type=int, default=15)
    parser.add_argument('--enable-anomaly-detection', action='store_true')
    args = parser.parse_args()

    config = vars(args)
    app = create_app(config)

    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
