"""
Fusion-ResNet NILM — Inference Server with Real-Time Smoothing
===============================================================

REST API that serves the Fusion-ResNet model with built-in temporal
smoothing. The mobile app does NOT need to smooth anything — just
call these endpoints:

For the mobile developer:
─────────────────────────
  POST /ingest    — Send a 400-sample window, get back stable status
                    (call this ~6 times/sec from the edge device)

  GET  /status    — Poll what's currently ON (cached, no model call)
                    (call this every 1-2 sec from the mobile app)

  GET  /timeline  — Get full session history (appliance ON/OFF events)
                    (call this when user opens "History" screen)

  POST /reset     — Clear session history (call on "New Session")

  GET  /health    — Server health check

Timing:
  The edge device should send one window every ~167ms (6/sec).
  The mobile app should poll /status every 1-2 seconds.
  /timeline can be called anytime — it returns everything since last /reset.

Usage:
    uvicorn deploy.serve:app --host 0.0.0.0 --port 8000
    docker build -t fusion-resnet-nilm . && docker run -p 8000:8000 fusion-resnet-nilm
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config (override via environment variables)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    str(PROJECT_ROOT / "checkpoints" / "best.pt"),
)
THRESHOLD = float(os.environ.get("MODEL_THRESHOLD", "0"))
DEVICE = os.environ.get("DEVICE", "cpu")
SMOOTH_WINDOW = int(os.environ.get("SMOOTH_WINDOW", "30"))   # ~5 sec at 6/sec
MIN_ON_COUNT = int(os.environ.get("MIN_ON_SECONDS", "3"))     # min 3s to report event

DEFAULT_NAMES = [
    'Air Conditioner', 'Blender', 'Coffee maker',
    'Compact Fluorescent Lamp', 'Fan', 'Fridge', 'Hair Iron',
    'Hairdryer', 'Heater', 'Incandescent Light Bulb', 'Laptop',
    'Microwave', 'Soldering Iron', 'Vacuum', 'Washing Machine',
]

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
threshold = 0.5
appliance_names: list[str] = []
n_classes = 0
startup_time = 0.0

# Real-time smoothing buffer + session history
_lock = threading.Lock()
_recent_preds: deque = deque(maxlen=SMOOTH_WINDOW)   # last N raw predictions
_current_status: dict = {}                             # smoothed "what's ON now"
_session_history: list = []                            # all predictions with timestamps
_session_start: float = 0.0


def load_model_once():
    global model, threshold, appliance_names, n_classes, startup_time, _session_start
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from fusion_resnet import FusionResNet

    t0 = time.time()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]

    cls_key = [k for k in state if "classifier" in k and "weight" in k][-1]
    n_classes = state[cls_key].shape[0]

    U = state["ica_branch.ica.U"].numpy()
    M = state["ica_branch.ica.M"].numpy()
    m = state["ica_branch.norm.m"].numpy()
    s = state["ica_branch.norm.s"].numpy()

    model = FusionResNet(
        n_classes=n_classes, signal_length=400,
        U=U, M=M, m=m, s=s,
    ).float()
    model.load_state_dict(state)
    model.eval()

    threshold = THRESHOLD if THRESHOLD > 0 else ckpt.get("threshold", 0.5)
    appliance_names = DEFAULT_NAMES[:n_classes]
    _session_start = time.time()

    with torch.no_grad():
        model(torch.randn(1, 400))

    startup_time = time.time() - t0
    print(f"Model loaded in {startup_time:.2f}s — "
          f"{n_classes} classes, threshold={threshold:.3f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_once()
    yield


app = FastAPI(
    title="Fusion-ResNet NILM",
    description="NILM appliance detection with real-time smoothing",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    signal: list[float] = Field(..., min_length=400, max_length=400,
                                description="400-sample current waveform window")

class StatusResponse(BaseModel):
    appliances_on: list[str] = Field(description="Appliances currently ON (smoothed)")
    confidence: dict[str, float] = Field(description="Smoothed confidence per appliance")
    session_duration_s: float
    windows_processed: int

class TimelineEvent(BaseModel):
    appliance: str
    state: str            # "ON" or "OFF"
    start_s: float        # seconds since session start
    end_s: float | None   # None if still ongoing
    duration_s: float

class TimelineResponse(BaseModel):
    events: list[TimelineEvent]
    session_duration_s: float
    summary: dict[str, float]   # appliance → total ON seconds

class HealthResponse(BaseModel):
    status: str
    model: str
    n_classes: int
    threshold: float
    smooth_window: int
    session_duration_s: float
    windows_processed: int


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_model(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(signal).float()
    logits = model(x)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    return preds[0], probs[0]


def _smooth_status() -> tuple[list[str], dict[str, float]]:
    """Majority-vote over the recent prediction buffer."""
    if not _recent_preds:
        return [], {n: 0.0 for n in appliance_names}

    arr = np.array(list(_recent_preds))       # (N, n_classes)
    vote_frac = arr.mean(axis=0)              # fraction of recent windows ON

    on_list = []
    conf = {}
    for i, name in enumerate(appliance_names):
        conf[name] = round(float(vote_frac[i]), 3)
        if vote_frac[i] >= 0.5:               # majority says ON
            on_list.append(name)

    return on_list, conf


def _build_timeline() -> tuple[list[dict], dict[str, float]]:
    """Scan session history and detect ON/OFF intervals per appliance."""
    if not _session_history:
        return [], {}

    all_preds = np.array([h["preds"] for h in _session_history])
    all_times = np.array([h["t"] for h in _session_history])
    n = len(all_preds)

    # Sliding-window majority-vote smooth
    kernel = min(SMOOTH_WINDOW, n)
    half = kernel // 2
    smoothed = np.zeros_like(all_preds)
    cumsum = np.vstack([np.zeros((1, n_classes)), np.cumsum(all_preds.astype(float), axis=0)])
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo) >= 0.5

    events = []
    summary = {}

    for c, name in enumerate(appliance_names):
        states = smoothed[:, c]
        diffs = np.diff(states.astype(int))

        on_starts = list(np.where(diffs == 1)[0] + 1)
        off_ends = list(np.where(diffs == -1)[0] + 1)

        if states[0] == 1:
            on_starts = [0] + on_starts
        if states[-1] == 1:
            off_ends.append(len(states))

        total_on = 0.0
        for si, ei in zip(on_starts, off_ends):
            t_start = float(all_times[si])
            if ei < len(all_times):
                t_end = float(all_times[ei])
            else:
                t_end = float(all_times[-1])

            dur = t_end - t_start
            if dur < MIN_ON_COUNT:
                continue

            total_on += dur
            events.append({
                "appliance": name,
                "state": "ON",
                "start_s": round(t_start, 1),
                "end_s": round(t_end, 1) if ei < len(all_times) else None,
                "duration_s": round(dur, 1),
            })

        if total_on > 0:
            summary[name] = round(total_on, 1)

    events.sort(key=lambda e: e["start_s"])
    return events, summary


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=StatusResponse)
def ingest(req: IngestRequest):
    """Edge device calls this ~6 times/sec with each new window.
    Returns the smoothed status (what's ON) immediately."""
    global _current_status

    signal = np.array(req.signal, dtype=np.float32).reshape(1, -1)
    preds, probs = _run_model(signal)
    now = time.time()
    t_session = now - _session_start

    with _lock:
        _recent_preds.append(preds)
        _session_history.append({"t": t_session, "preds": preds.tolist()})
        on_list, conf = _smooth_status()
        _current_status = {"on": on_list, "conf": conf}

    return StatusResponse(
        appliances_on=on_list,
        confidence=conf,
        session_duration_s=round(t_session, 1),
        windows_processed=len(_session_history),
    )


@app.get("/status", response_model=StatusResponse)
def status():
    """Mobile app polls this every 1-2 sec. Returns cached smoothed state.
    No model call — instant response."""
    with _lock:
        on_list = _current_status.get("on", [])
        conf = _current_status.get("conf", {n: 0.0 for n in appliance_names})
        n_windows = len(_session_history)

    return StatusResponse(
        appliances_on=on_list,
        confidence=conf,
        session_duration_s=round(time.time() - _session_start, 1),
        windows_processed=n_windows,
    )


@app.get("/timeline", response_model=TimelineResponse)
def timeline():
    """Returns all ON/OFF events since session start.
    Call when user opens the history/report screen."""
    with _lock:
        events, summary = _build_timeline()

    return TimelineResponse(
        events=[TimelineEvent(**e) for e in events],
        session_duration_s=round(time.time() - _session_start, 1),
        summary=summary,
    )


@app.post("/reset")
def reset():
    """Start a new monitoring session. Clears all history."""
    global _session_start, _current_status
    with _lock:
        _recent_preds.clear()
        _session_history.clear()
        _current_status = {}
        _session_start = time.time()
    return {"status": "ok", "message": "Session reset"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model="FusionResNet",
        n_classes=n_classes,
        threshold=threshold,
        smooth_window=SMOOTH_WINDOW,
        session_duration_s=round(time.time() - _session_start, 1),
        windows_processed=len(_session_history),
    )
