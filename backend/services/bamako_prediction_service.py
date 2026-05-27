"""
Service de prédiction pour Bamako.
Charge le modèle LSTM spécifique, les séquences les plus récentes par commune
et fournit une API simplifiée pour exposer ces prédictions au frontend.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np

from models.lstm_model_bamako import LSTMPredictorBamako
from services.bamako_live_sequence_builder import (
    build_model_sequence,
    fetch_merged_daily_meteo,
)
from utils.bamako_communes import BAMAKO_COMMUNES
from utils.bamako_features import get_risk_factors, get_static_features
from utils.bamako_neighborhoods import (
    BAMAKO_COMMUNE_NEIGHBORHOODS,
    get_commune_from_neighborhood,
    normalize_commune_name,
)


@dataclass
class CommuneSequence:
    commune: str
    end_date: Optional[str]
    split: str
    sequence: np.ndarray


class BamakoPredictionService:
    """Chargement du modèle Bamako + expose des prédictions par commune/quartier."""

    def __init__(self):
        backend_root = os.path.dirname(os.path.dirname(__file__))
        self.training_dir = os.path.join(backend_root, "data", "training", "bamako_lstm")
        self.lstm_predictor = LSTMPredictorBamako()
        self.sequences_by_commune: Dict[str, CommuneSequence] = {}
        self.last_loaded_at: Optional[datetime] = None
        self._load_latest_sequences()
        # Fit scaler on loaded sequences to avoid "not fitted" errors at inference
        try:
            self._fit_scaler_with_sequences()
        except Exception as exc:
            print(f"[BamakoPredictionService] ⚠️ Impossible de fitter le scaler: {exc}")

    def _fit_scaler_with_sequences(self):
        """Fit the LSTM scaler using loaded commune sequences (fallback if scaler.pkl absent)."""
        if not self.sequences_by_commune:
            return
        sequences = np.stack([seq.sequence for seq in self.sequences_by_commune.values()], axis=0)
        flat = sequences.reshape(-1, sequences.shape[-1])
        if np.isnan(flat).all():
            return
        self.lstm_predictor.scaler.fit(flat)
        # Persist scaler for next runs
        try:
            joblib.dump(self.lstm_predictor.scaler, self.lstm_predictor.scaler_path)
        except Exception as exc:
            print(f"[BamakoPredictionService] ⚠️ Impossible de sauvegarder le scaler: {exc}")

    # ------------------------------------------------------------------ #
    # Chargement des séquences depuis les fichiers numpy/metadata
    # ------------------------------------------------------------------ #
    def _load_latest_sequences(self):
        if not os.path.isdir(self.training_dir):
            print(f"[BamakoPredictionService] ⚠️ Dossier introuvable : {self.training_dir}")
            return

        sequences: Dict[str, CommuneSequence] = {}
        for split in ("train", "val"):
            x_path = os.path.join(self.training_dir, f"X_{split}.npy")
            meta_path = os.path.join(self.training_dir, f"metadata_{split}.json")
            if not os.path.exists(x_path) or not os.path.exists(meta_path):
                continue

            try:
                data = np.load(x_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                print(f"[BamakoPredictionService] ❌ Impossible de charger {split}: {exc}")
                continue

            if len(metadata) != len(data):
                print(
                    f"[BamakoPredictionService] ⚠️ Taille metadata ({len(metadata)}) ≠ données ({len(data)}) pour {split}"
                )

            for idx, meta in enumerate(metadata):
                if idx >= len(data):
                    break
                commune = meta.get("commune")
                end_date = meta.get("end_date")
                if not commune:
                    continue

                try:
                    sequence = np.array(data[idx], dtype=np.float32, copy=True)
                except Exception as exc:
                    print(f"[BamakoPredictionService] ⚠️ Impossible de copier la séquence {idx}: {exc}")
                    continue

                key = commune
                current = sequences.get(key)

                def to_dt(value: Optional[str]) -> datetime:
                    if not value:
                        return datetime.min
                    try:
                        return datetime.strptime(value, "%Y-%m-%d")
                    except ValueError:
                        return datetime.min

                new_dt = to_dt(end_date)
                current_dt = to_dt(current.end_date) if current else datetime.min

                if new_dt >= current_dt:
                    sequences[key] = CommuneSequence(
                        commune=commune,
                        end_date=end_date,
                        split=split,
                        sequence=sequence,
                    )

        self.sequences_by_commune = sequences
        self.last_loaded_at = datetime.now()
        print(
            f"[BamakoPredictionService] ✅ Séquences chargées pour {len(self.sequences_by_commune)} communes "
            f"(dernière mise à jour: {self.last_loaded_at.isoformat(timespec='seconds')})"
        )

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #
    def _model_n_features(self) -> Optional[int]:
        model = self.lstm_predictor.model
        if model is not None and getattr(model, "input_shape", None):
            shape = model.input_shape
            if shape and len(shape) >= 3 and shape[-1]:
                return int(shape[-1])
        return None

    def predict(
        self,
        commune: Optional[str] = None,
        neighborhood: Optional[str] = None,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        use_live_weather: bool = True,
    ) -> Dict:
        """Prédiction Bamako : météo live par défaut, repli sur séquences .npy."""
        if use_live_weather:
            try:
                return self.predict_live(
                    commune=commune,
                    neighborhood=neighborhood,
                    latitude=latitude,
                    longitude=longitude,
                )
            except Exception as exc:
                print(
                    f"[BamakoPredictionService] ⚠️ predict_live échoué ({exc}), "
                    "repli séquences entraînement"
                )
        return self._predict_from_cached_sequences(commune=commune, neighborhood=neighborhood)

    def predict_live(
        self,
        commune: Optional[str] = None,
        neighborhood: Optional[str] = None,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict:
        """
        Reconstruit une fenêtre de 30 jours (pas journalier) avec météo récente
        puis infère avec lstm_model_bamako.h5.
        """
        start_time = time.perf_counter()
        resolved_commune = self._resolve_commune(commune, neighborhood)
        if not resolved_commune:
            raise ValueError("Commune ou quartier inconnu. Précisez 'commune' ou 'neighborhood'.")

        commune_info = BAMAKO_COMMUNES.get(resolved_commune, {})
        if latitude is not None and longitude is not None:
            lat, lon = float(latitude), float(longitude)
        else:
            lat = commune_info.get("lat")
            lon = commune_info.get("lon")
        if lat is None or lon is None:
            raise ValueError(f"Coordonnées manquantes pour {resolved_commune}")

        seq_len = self.lstm_predictor.sequence_length
        daily_df, weather_meta = fetch_merged_daily_meteo(
            float(lat),
            float(lon),
            sequence_length=seq_len,
            forecast_days=self.lstm_predictor.forecast_days,
        )
        sequence, feature_cols = build_model_sequence(
            daily_df,
            resolved_commune,
            seq_len,
            target_n_features=self._model_n_features(),
        )
        batch = sequence[np.newaxis, ...]
        probability = float(self.lstm_predictor.predict(batch)[0])
        risk_level = self.lstm_predictor._get_risk_level(probability)

        static_features = get_static_features(resolved_commune) or {}
        risk_factors = get_risk_factors(resolved_commune) or {}
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        last_row = daily_df.iloc[-1]
        return {
            "commune": resolved_commune,
            "neighborhood": neighborhood,
            "prediction": {
                "flood_probability": probability,
                "risk_level": risk_level,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sequence_end_date": weather_meta.get("sequence_end_date"),
                "sequence_start_date": weather_meta.get("sequence_start_date"),
                "last_day_precipitation_mm": float(last_row.get("precipitation", 0)),
                "antecedent_precip_7d_mm": float(last_row.get("antecedent_precip_7d", 0)),
                "coordinates": {"lat": lat, "lon": lon},
                "source": "lstm_bamako_live",
                "stale": False,
            },
            "context": {
                "static_features": static_features,
                "risk_factors": risk_factors,
                "zones_risque": commune_info.get("zones_risque"),
            },
            "metadata": {
                "latency_ms": duration_ms,
                "inference_mode": "live_weather",
                "data_sources": weather_meta.get("data_sources", []),
                "feature_count": len(feature_cols),
                "features_used": feature_cols,
                "weather_error": weather_meta.get("weather_error"),
                "forecast_days_available": weather_meta.get("forecast_days_available"),
            },
        }

    def _predict_from_cached_sequences(
        self,
        commune: Optional[str] = None,
        neighborhood: Optional[str] = None,
    ) -> Dict:
        start_time = time.perf_counter()
        resolved_commune = self._resolve_commune(commune, neighborhood)
        if not resolved_commune:
            raise ValueError("Commune ou quartier inconnu. Précisez 'commune' ou 'neighborhood'.")

        sequence_info = self.sequences_by_commune.get(resolved_commune)
        if not sequence_info:
            raise ValueError(f"Aucune séquence disponible pour {resolved_commune}.")

        sequence = sequence_info.sequence[np.newaxis, ...]
        probability = float(self.lstm_predictor.predict(sequence)[0])
        risk_level = self.lstm_predictor._get_risk_level(probability)

        commune_info = BAMAKO_COMMUNES.get(resolved_commune, {})
        static_features = get_static_features(resolved_commune) or {}
        risk_factors = get_risk_factors(resolved_commune) or {}

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "commune": resolved_commune,
            "neighborhood": neighborhood,
            "prediction": {
                "flood_probability": probability,
                "risk_level": risk_level,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sequence_end_date": sequence_info.end_date,
                "sequence_split": sequence_info.split,
                "coordinates": commune_info.get("lat") and commune_info.get("lon")
                and {"lat": commune_info["lat"], "lon": commune_info["lon"]}
                or None,
                "source": "lstm_bamako_cached",
                "stale": True,
            },
            "context": {
                "static_features": static_features,
                "risk_factors": risk_factors,
                "zones_risque": commune_info.get("zones_risque"),
            },
            "metadata": {
                "latency_ms": duration_ms,
                "inference_mode": "cached_npy",
                "last_sequence_loaded_at": self.last_loaded_at.isoformat(timespec="seconds")
                if self.last_loaded_at
                else None,
            },
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_commune(self, commune: Optional[str], neighborhood: Optional[str]) -> Optional[str]:
        if neighborhood:
            resolved = get_commune_from_neighborhood(neighborhood)
            if resolved:
                return resolved

        normalized_commune = normalize_commune_name(commune) if commune else None
        if normalized_commune:
            return normalized_commune

        # Si quartier inconnu mais ressemble à "Commune X"
        if neighborhood:
            normalized_commune = normalize_commune_name(neighborhood)
            if normalized_commune:
                return normalized_commune

        return None

    def list_supported_neighborhoods(self) -> Dict[str, list]:
        return BAMAKO_COMMUNE_NEIGHBORHOODS


