"""
Construit une séquence LSTM Bamako (30 jours) à partir de météo récente + prévisions OpenWeather.
Utilisé à l'inférence pour remplacer les séquences figées (.npy).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.africa_data_service import AfricaDataService
from services.weather_forecast_service import WeatherForecastService
from utils.bamako_features import get_static_features

TEMPORAL_FEATURE_COLS = [
    "precipitation",
    "temperature",
    "humidity",
    "pressure",
    "soil_moisture",
    "antecedent_precip_3d",
    "antecedent_precip_7d",
    "antecedent_precip_14d",
    "soil_saturation_index",
]

STATIC_FEATURE_COLS = [
    "elevation",
    "slope",
    "distance_to_river",
    "in_flood_plain",
    "depression_depth",
    "drainage_density",
    "drainage_state",
    "drainage_coverage",
    "blocked_drainage_pct",
    "impermeable_surface",
    "building_density",
    "population_density",
    "groundwater_level",
    "runoff_coefficient",
    "infiltration_rate",
    "soil_permeability",
    "vegetation_cover",
    "ndvi_avg",
    "informal_settlement_pct",
    "poverty_index",
    "flood_preparedness",
]


def _parse_date_key(value: str) -> str:
    if not value:
        return ""
    return str(value)[:10]


def fetch_merged_daily_meteo(
    lat: float,
    lon: float,
    sequence_length: int = 30,
    forecast_days: int = 7,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Série journalière sur `sequence_length` jours se terminant aujourd'hui.
    Précipitations : CHIRPS (AfricaDataService) + enrichissement OpenWeather.
    """
    africa = AfricaDataService()
    weather = WeatherForecastService()

    end_day = date.today()
    start_day = end_day - timedelta(days=sequence_length - 1)
    meteo_bundle = africa.get_comprehensive_meteo_data(lat, lon, days_back=sequence_length)

    precip_by_date: Dict[str, float] = {}
    for entry in meteo_bundle.get("chirps", {}).get("daily_data", []) or []:
        key = _parse_date_key(entry.get("date", ""))
        if key:
            precip_by_date[key] = float(entry.get("precipitation", 0) or 0)

    ow_by_date: Dict[str, Dict] = {}
    data_sources = ["chirps_simulated_or_cached"]
    weather_error: Optional[str] = None

    try:
        forecast = weather.get_forecast(lat, lon, days=forecast_days)
        if forecast.get("source") != "fallback":
            data_sources.append("openweather_forecast")
        for day in forecast.get("forecasts", []) or []:
            key = _parse_date_key(day.get("date", ""))
            if key:
                ow_by_date[key] = day
    except Exception as exc:
        weather_error = str(exc)

    try:
        current = weather.get_current_weather(lat, lon)
        today_key = end_day.isoformat()
        if current and current.get("source") != "fallback":
            data_sources.append("openweather_current")
            ow_by_date[today_key] = {
                "date": today_key,
                "temperature": {"avg": current.get("temperature", 30.0)},
                "humidity": {"avg": current.get("humidity", 60.0)},
                "pressure": {"avg": current.get("pressure", 1013.0)},
                "precipitation": {"total": current.get("precipitation", 0.0)},
            }
    except Exception as exc:
        if not weather_error:
            weather_error = str(exc)

    rows: List[Dict] = []
    cursor = start_day
    while cursor <= end_day:
        key = cursor.isoformat()
        precip = precip_by_date.get(key, 0.0)
        ow = ow_by_date.get(key, {})

        temp = 30.0
        humidity = 60.0
        pressure = 1013.0

        if ow:
            temps = ow.get("temperature") or {}
            if isinstance(temps, dict):
                temp = float(temps.get("avg") or temps.get("max") or temp)
            hum = ow.get("humidity") or {}
            if isinstance(hum, dict):
                humidity = float(hum.get("avg") or humidity)
            pres = ow.get("pressure") or {}
            if isinstance(pres, dict):
                pressure = float(pres.get("avg") or pressure)
            precip_ow = ow.get("precipitation") or {}
            if isinstance(precip_ow, dict):
                precip = float(precip_ow.get("total") or precip)
            elif isinstance(precip_ow, (int, float)):
                precip = float(precip_ow)

        rows.append(
            {
                "date": pd.Timestamp(cursor),
                "precipitation": max(0.0, precip),
                "temperature": temp,
                "humidity": np.clip(humidity, 0.0, 100.0),
                "pressure": pressure,
            }
        )
        cursor += timedelta(days=1)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Impossible de construire la série météo journalière")

    df = df.sort_values("date").reset_index(drop=True)

    precip_series = df["precipitation"].astype(float)
    df["soil_moisture"] = np.clip(
        0.25 + (precip_series.rolling(window=7, min_periods=1).sum() / 120.0),
        0.0,
        1.0,
    )
    df["antecedent_precip_3d"] = precip_series.rolling(window=3, min_periods=1).sum()
    df["antecedent_precip_7d"] = precip_series.rolling(window=7, min_periods=1).sum()
    df["antecedent_precip_14d"] = precip_series.rolling(window=14, min_periods=1).sum()
    df["soil_saturation_index"] = np.clip(precip_series.rolling(window=30, min_periods=1).sum() / 200.0, 0.0, 1.0)

    meta = {
        "sequence_start_date": start_day.isoformat(),
        "sequence_end_date": end_day.isoformat(),
        "data_sources": list(dict.fromkeys(data_sources)),
        "weather_error": weather_error,
        "forecast_days_available": forecast_days,
    }
    return df, meta


def build_model_sequence(
    daily_df: pd.DataFrame,
    commune: str,
    sequence_length: int,
    target_n_features: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble (sequence_length, n_features) = temporel + statique répété."""
    if len(daily_df) < sequence_length:
        raise ValueError(
            f"Série trop courte ({len(daily_df)} jours, besoin de {sequence_length})"
        )

    window = daily_df.iloc[-sequence_length:].copy()

    temporal_cols = [c for c in TEMPORAL_FEATURE_COLS if c in window.columns]
    static_features = get_static_features(commune) or {}
    static_cols = [c for c in STATIC_FEATURE_COLS if c in static_features]

    temporal_seq = window[temporal_cols].astype(np.float32).values
    if static_cols:
        static_vals = np.array([float(static_features[c]) for c in static_cols], dtype=np.float32)
        static_seq = np.tile(static_vals, (sequence_length, 1))
        sequence = np.concatenate([temporal_seq, static_seq], axis=1)
        used_cols = temporal_cols + static_cols
    else:
        sequence = temporal_seq
        used_cols = temporal_cols

    sequence = np.nan_to_num(sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if target_n_features is not None and sequence.shape[1] != target_n_features:
        if sequence.shape[1] > target_n_features:
            sequence = sequence[:, :target_n_features]
            used_cols = used_cols[:target_n_features]
        else:
            pad = np.zeros((sequence_length, target_n_features - sequence.shape[1]), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=1)
            used_cols = used_cols + [f"pad_{i}" for i in range(pad.shape[1])]

    return sequence, used_cols
