"""
F1 Tire Strategy API
====================
FastAPI backend that wraps the Phase 4 TireStrategyPredictor.
Deploy on Railway or Render (both free tier).

Setup:
    pip install fastapi uvicorn torch scikit-learn pandas fastf1

Run locally:
    uvicorn strategy_api:app --reload --port 8000

Deploy to Railway:
    1. railway init
    2. railway up
    3. Set env var: PORT (Railway sets this automatically)

File layout expected:
    strategy_api.py          ← this file
    best_model_phase3.pt     ← your trained model weights
    requirements.txt         ← pip dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import fastf1
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

app = FastAPI(title="F1 Tire Strategy API", version="1.0.0")

# ── CORS — allow your Vercel PWA to call this API ─────────────────────────
# Replace with your actual Vercel URL once deployed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to ["https://f1app-seven.vercel.app"] in prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION (identical to Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

class TireDegradationLSTMv2(nn.Module):
    def __init__(self, input_size, num_tracks, hidden_size=128,
                 num_layers=2, dropout=0.3, track_embed_dim=8):
        super().__init__()
        self.track_embedding = nn.Embedding(num_tracks, track_embed_dim)
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout
        )
        head_input_size = hidden_size + track_embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),             nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, track_id):
        lstm_out, _ = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]
        track_emb    = self.track_embedding(track_id)
        combined     = torch.cat([last_hidden, track_emb], dim=1)
        return self.head(combined)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "TyreLife", "StintLap", "TrackTemp", "AirTemp", "FuelProxy",
    "PositionNorm", "StintProgress",
    "Compound_SOFT", "Compound_MEDIUM", "Compound_HARD",
    "Sector1TimeSec", "Sector2TimeSec", "Sector3TimeSec",
]
WINDOW_SIZE = 10

# Track names the model was trained on — must match FastF1 race names exactly
KNOWN_TRACKS = [
    "Bahrain", "Saudi Arabia", "Australia", "Imola", "Spain", "Monaco",
    "Canada", "Britain", "Austria", "France", "Hungary", "Belgium",
    "Italy", "Singapore", "Japan", "Mexico", "Brazil", "Abu Dhabi",
    "Azerbaijan", "Miami", "Qatar",
]

# Default sector times per track (approximate, seconds)
# Used during simulation since we don't have future sector data
TRACK_SECTOR_DEFAULTS = {
    "Bahrain":      (28.0, 38.0, 27.0),
    "Saudi Arabia": (29.0, 37.0, 24.0),
    "Australia":    (26.0, 39.0, 23.0),
    "Spain":        (27.0, 42.0, 22.0),
    "Monaco":       (25.0, 32.0, 23.0),
    "Canada":       (26.0, 38.0, 25.0),
    "Britain":      (27.0, 40.0, 23.0),
    "Hungary":      (28.0, 38.0, 22.0),
    "Italy":        (23.0, 35.0, 21.0),
    "Japan":        (30.0, 42.0, 23.0),
    "Abu Dhabi":    (27.0, 38.0, 22.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — LOAD MODEL AND FIT SCALER
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cpu")  # CPU on server — GPU not needed for inference

# FastF1 cache
os.makedirs("f1_cache_api", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache_api")

print("Loading training data to refit scaler...")

RACES_2022 = [
    (2022, "Bahrain"), (2022, "Saudi Arabia"), (2022, "Australia"),
    (2022, "Imola"),   (2022, "Spain"),        (2022, "Monaco"),
    (2022, "Canada"),  (2022, "Britain"),       (2022, "Austria"),
    (2022, "France"),  (2022, "Hungary"),       (2022, "Belgium"),
    (2022, "Italy"),   (2022, "Singapore"),     (2022, "Japan"),
    (2022, "Mexico"),  (2022, "Brazil"),        (2022, "Abu Dhabi"),
]
RACES_2023 = [
    (2023, "Bahrain"), (2023, "Saudi Arabia"), (2023, "Australia"),
    (2023, "Azerbaijan"), (2023, "Miami"),     (2023, "Monaco"),
    (2023, "Spain"),   (2023, "Canada"),       (2023, "Britain"),
    (2023, "Hungary"), (2023, "Belgium"),      (2023, "Italy"),
    (2023, "Singapore"), (2023, "Japan"),      (2023, "Qatar"),
    (2023, "Mexico"),  (2023, "Brazil"),       (2023, "Abu Dhabi"),
]
ALL_RACES = RACES_2022 + RACES_2023


def load_race_for_scaler(year, race_name):
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        all_laps = session.laps

        def is_clean_lap(lap):
            return not any([
                pd.notna(lap["PitInTime"]), pd.notna(lap["PitOutTime"]),
                lap["TrackStatus"] not in ["1", ""], pd.isna(lap["LapTime"]),
            ])

        clean = all_laps[all_laps.apply(is_clean_lap, axis=1)].copy()
        if len(clean) < 50:
            return None

        clean["LapTimeSeconds"] = clean["LapTime"].dt.total_seconds()
        clean["Season"]   = year
        clean["RaceName"] = race_name
        clean = clean.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

        def assign_stints(d):
            d = d.copy()
            tl = d["TyreLife"]
            d["StintNumber"] = (tl <= tl.shift(1, fill_value=999)).cumsum() + 1
            return d
        clean = clean.groupby("Driver", group_keys=False).apply(assign_stints)

        def add_delta(s):
            s = s.copy()
            s["LapTimeDelta"] = s["LapTimeSeconds"] - s["LapTimeSeconds"].iloc[0]
            s["StintLap"] = range(1, len(s) + 1)
            return s
        clean = clean.groupby(["Driver", "StintNumber"], group_keys=False).apply(add_delta)

        weather = session.laps.get_weather_data()[["Time", "TrackTemp", "AirTemp"]].copy()
        weather = weather.rename(columns={"Time": "WeatherTime"}).sort_values("WeatherTime")
        clean = pd.merge_asof(
            clean.sort_values("Time"), weather,
            left_on="Time", right_on="WeatherTime", direction="nearest"
        )

        race_laps = clean["LapNumber"].max()
        clean["FuelProxy"]    = 1.0 - (clean["LapNumber"] / race_laps)
        clean["PositionNorm"] = clean["Position"] / clean["Driver"].nunique()
        stint_lengths = clean.groupby(["Driver", "StintNumber"])["StintLap"].transform("max")
        clean["StintProgress"] = clean["StintLap"] / stint_lengths

        for col in ["Compound_SOFT", "Compound_MEDIUM", "Compound_HARD"]:
            clean[col] = 0
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            clean.loc[clean["Compound"] == compound, f"Compound_{compound}"] = 1

        for s in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            if s in clean.columns:
                clean[f"{s}Sec"] = clean[s].dt.total_seconds()
            else:
                clean[f"{s}Sec"] = 28.0  # fallback

        stint_max = clean.groupby(["Driver", "StintNumber"])["LapTimeDelta"].transform("max")
        stint_min = clean.groupby(["Driver", "StintNumber"])["LapTimeDelta"].transform("min")
        clean = clean[(stint_max <= 15.0) & (stint_min >= -15.0)]

        return clean
    except Exception as e:
        print(f"  Failed {year} {race_name}: {e}")
        return None


# Load training data and fit scaler + encoder at startup
frames = []
for year, name in ALL_RACES:
    r = load_race_for_scaler(year, name)
    if r is not None:
        frames.append(r)

train_data = pd.concat(frames, ignore_index=True)
scaler = StandardScaler()
scaler.fit(train_data[FEATURE_COLS].dropna().values.astype(np.float32))

track_encoder = LabelEncoder()
track_encoder.fit(train_data["RaceName"].values)
num_tracks = len(track_encoder.classes_)

# Load model weights
MODEL_PATH = "best_model_phase3.pt"
assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found — run Phase 3 training first"

model = TireDegradationLSTMv2(
    input_size=len(FEATURE_COLS), num_tracks=num_tracks,
    hidden_size=128, num_layers=2, dropout=0.3, track_embed_dim=8
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Model loaded. Tracks known: {num_tracks}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR CLASS (same as Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

class TireStrategyPredictor:
    def __init__(self, model, scaler, track_encoder, feature_cols, window_size, device):
        self.model         = model
        self.scaler        = scaler
        self.track_encoder = track_encoder
        self.feature_cols  = feature_cols
        self.window_size   = window_size
        self.device        = device

    def _encode_track(self, track_name):
        known = list(self.track_encoder.classes_)
        if track_name not in known:
            raise ValueError(f"Unknown track: {track_name}. Known: {known}")
        return int(self.track_encoder.transform([track_name])[0])

    def _build_feature_row(self, stint_lap, tyre_life, track_temp, air_temp,
                           fuel_proxy, position_norm, stint_progress, compound,
                           s1=28.0, s2=38.0, s3=27.0):
        row = {
            "TyreLife": tyre_life, "StintLap": stint_lap,
            "TrackTemp": track_temp, "AirTemp": air_temp,
            "FuelProxy": fuel_proxy, "PositionNorm": position_norm,
            "StintProgress": stint_progress,
            "Compound_SOFT":   1 if compound == "SOFT"   else 0,
            "Compound_MEDIUM": 1 if compound == "MEDIUM" else 0,
            "Compound_HARD":   1 if compound == "HARD"   else 0,
            "Sector1TimeSec": s1, "Sector2TimeSec": s2, "Sector3TimeSec": s3,
        }
        return np.array([row[c] for c in self.feature_cols], dtype=np.float32)

    def predict_next_lap(self, window_features, track_name):
        scaled = self.scaler.transform(window_features)
        X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        track_id = torch.tensor(
            [self._encode_track(track_name)], dtype=torch.long
        ).to(self.device)
        with torch.no_grad():
            return self.model(X, track_id).item()

    def simulate_full_stint(self, track_name, compound, total_race_laps,
                            current_race_lap, tyre_age_at_start,
                            track_temp, air_temp, driver_position,
                            total_drivers=20, max_stint_laps=35):
        s1, s2, s3 = TRACK_SECTOR_DEFAULTS.get(track_name, (28.0, 38.0, 27.0))
        results = []
        window  = []

        for stint_lap in range(1, max_stint_laps + 1):
            race_lap      = current_race_lap + stint_lap - 1
            tyre_life     = tyre_age_at_start + stint_lap
            fuel_proxy    = max(0.0, 1.0 - (race_lap / total_race_laps))
            position_norm = driver_position / total_drivers
            stint_progress = stint_lap / max_stint_laps

            # Add slight noise to sector times to prevent autoregressive drift
            row = self._build_feature_row(
                stint_lap, tyre_life, track_temp, air_temp,
                fuel_proxy, position_norm, stint_progress, compound,
                s1=s1 + np.random.normal(0, 0.15),
                s2=s2 + np.random.normal(0, 0.15),
                s3=s3 + np.random.normal(0, 0.15),
            )
            window.append(row)

            if len(window) < self.window_size:
                padded = [window[0]] * (self.window_size - len(window)) + window
                window_arr = np.array(padded, dtype=np.float32)
            else:
                window_arr = np.array(window[-self.window_size:], dtype=np.float32)

            predicted_delta = self.predict_next_lap(window_arr, track_name)
            results.append({
                "stint_lap":       stint_lap,
                "tyre_life":       tyre_life,
                "race_lap":        race_lap,
                "predicted_delta": round(predicted_delta, 3),
                "compound":        compound,
            })

        return results


predictor = TireStrategyPredictor(
    model, scaler, track_encoder, FEATURE_COLS, WINDOW_SIZE, device
)


def compute_pit_window(stint_data, deg_rate_threshold=0.08, min_stint_laps=8):
    deltas = [d["predicted_delta"] for d in stint_data]
    rates  = []
    for i in range(len(deltas)):
        if i < 3:
            rates.append(0.0)
        else:
            rates.append(np.mean(np.diff(deltas[max(0,i-3):i+1])))

    pit_laps = [
        d["stint_lap"] for d, r in zip(stint_data, rates)
        if abs(r) >= deg_rate_threshold and d["stint_lap"] >= min_stint_laps
    ]

    if not pit_laps:
        return {"window_start": None, "window_end": None, "optimal_lap": None,
                "reason": "Tires holding — no pit recommended in this window"}

    optimal = pit_laps[0]
    return {
        "window_start": max(min_stint_laps, optimal - 2),
        "window_end":   optimal + 3,
        "optimal_lap":  optimal,
        "reason":       f"Degradation rate exceeded threshold at stint lap {optimal}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

class StrategyRequest(BaseModel):
    track: str                        # e.g. "Bahrain"
    compound: str                     # "SOFT", "MEDIUM", or "HARD"
    total_race_laps: int              # total laps in the race
    current_race_lap: int = 1         # race lap number when stint starts
    tyre_age_at_start: int = 0        # laps already on this set (0 = fresh)
    track_temp: float = 38.0          # degrees C
    air_temp: float = 28.0            # degrees C
    driver_position: int = 1          # current race position
    total_drivers: int = 20
    max_stint_laps: int = 35          # how many laps to simulate
    deg_rate_threshold: float = 0.08  # s/lap to trigger pit recommendation


class CompoundComparison(BaseModel):
    track: str
    total_race_laps: int
    current_race_lap: int = 1
    tyre_age_at_start: int = 0
    track_temp: float = 38.0
    air_temp: float = 28.0
    driver_position: int = 1
    total_drivers: int = 20
    max_stint_laps: int = 35
    deg_rate_threshold: float = 0.08


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "F1 Tire Degradation LSTM v2",
        "known_tracks": KNOWN_TRACKS,
        "endpoints": ["/strategy", "/compare", "/tracks", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/tracks")
def tracks():
    """Return the list of tracks the model knows about."""
    return {"tracks": sorted(KNOWN_TRACKS)}


@app.post("/strategy")
def strategy(req: StrategyRequest):
    """
    Simulate a full stint and return degradation curve + pit window.

    Example request body:
    {
        "track": "Bahrain",
        "compound": "SOFT",
        "total_race_laps": 57,
        "current_race_lap": 1,
        "tyre_age_at_start": 0,
        "track_temp": 38,
        "air_temp": 28,
        "driver_position": 1
    }
    """
    if req.track not in KNOWN_TRACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown track '{req.track}'. Known tracks: {sorted(KNOWN_TRACKS)}"
        )
    if req.compound not in ["SOFT", "MEDIUM", "HARD"]:
        raise HTTPException(
            status_code=400,
            detail="compound must be SOFT, MEDIUM, or HARD"
        )

    try:
        stint_data = predictor.simulate_full_stint(
            track_name        = req.track,
            compound          = req.compound,
            total_race_laps   = req.total_race_laps,
            current_race_lap  = req.current_race_lap,
            tyre_age_at_start = req.tyre_age_at_start,
            track_temp        = req.track_temp,
            air_temp          = req.air_temp,
            driver_position   = req.driver_position,
            total_drivers     = req.total_drivers,
            max_stint_laps    = req.max_stint_laps,
        )
        pit_window = compute_pit_window(stint_data, req.deg_rate_threshold)

        return {
            "track":      req.track,
            "compound":   req.compound,
            "pit_window": pit_window,
            "stint_data": stint_data,
            "summary": {
                "delta_at_lap_10": next(
                    (d["predicted_delta"] for d in stint_data if d["stint_lap"] == 10), None
                ),
                "delta_at_lap_20": next(
                    (d["predicted_delta"] for d in stint_data if d["stint_lap"] == 20), None
                ),
                "total_degradation": round(
                    max(d["predicted_delta"] for d in stint_data) -
                    min(d["predicted_delta"] for d in stint_data), 3
                ),
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/compare")
def compare(req: CompoundComparison):
    """
    Simulate all three compounds for the same scenario and return a comparison.
    Use this to power the Strategy tab compound selector in the PWA.
    """
    if req.track not in KNOWN_TRACKS:
        raise HTTPException(status_code=400, detail=f"Unknown track '{req.track}'")

    results = {}
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        stint_data = predictor.simulate_full_stint(
            track_name        = req.track,
            compound          = compound,
            total_race_laps   = req.total_race_laps,
            current_race_lap  = req.current_race_lap,
            tyre_age_at_start = req.tyre_age_at_start,
            track_temp        = req.track_temp,
            air_temp          = req.air_temp,
            driver_position   = req.driver_position,
            total_drivers     = req.total_drivers,
            max_stint_laps    = req.max_stint_laps,
        )
        pit_window = compute_pit_window(stint_data, req.deg_rate_threshold)
        results[compound] = {
            "pit_window": pit_window,
            "stint_data": stint_data,
            "total_degradation": round(
                max(d["predicted_delta"] for d in stint_data) -
                min(d["predicted_delta"] for d in stint_data), 3
            ),
        }

    return {
        "track":    req.track,
        "scenario": req.dict(exclude={"track"}),
        "compounds": results,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("strategy_api:app", host="0.0.0.0", port=port)
