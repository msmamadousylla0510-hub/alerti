"""
Weather Forecast Service - Fetches current weather and forecasts from OpenWeatherMap
"""
import os
from pathlib import Path

import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

from utils.config import OPENWEATHERMAP_API_KEY

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# Cache court pour le menu radial (évite 429 One Call à chaque clic)
_SNAPSHOT_CACHE: Dict[str, Tuple[datetime, Dict]] = {}
_SNAPSHOT_CACHE_TTL = timedelta(minutes=10)


class WeatherForecastService:
    """Service for fetching weather forecasts from OpenWeatherMap"""
    
    def __init__(self):
        self.base_url_v3 = "https://api.openweathermap.org/data/3.0"
        self.one_call_url_v3 = f"{self.base_url_v3}/onecall"
        self.current_weather_url = "https://api.openweathermap.org/data/2.5/weather"
        self.air_pollution_url = "https://api.openweathermap.org/data/2.5/air_pollution"

    @staticmethod
    def _read_key_from_env_file() -> str:
        if not _ENV_FILE.is_file():
            return ""
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("OPENWEATHERMAP_API_KEY="):
                return stripped.split("=", 1)[1].strip().strip('"').strip("'")
        return ""

    @property
    def api_key(self) -> str:
        """Toujours la clé de alerti/.env (pas une variable shell obsolète)."""
        load_dotenv(_ENV_FILE, override=True)
        return (os.getenv("OPENWEATHERMAP_API_KEY") or self._read_key_from_env_file() or "").strip()

    def key_diagnostics(self) -> Dict:
        """Vérifie que la clé chargée correspond au .env et teste One Call 3.0."""
        import hashlib

        file_key = self._read_key_from_env_file()
        runtime_key = self.api_key
        file_hash = hashlib.sha256(file_key.encode()).hexdigest()[:12] if file_key else ""
        runtime_hash = hashlib.sha256(runtime_key.encode()).hexdigest()[:12] if runtime_key else ""

        test_status = None
        test_message = None
        if runtime_key:
            try:
                response = self._request_openweather(
                    self.one_call_url_v3,
                    {"lat": 12.65, "lon": -7.98, "exclude": "minutely,daily,alerts"},
                )
                test_status = response.status_code
                if response.status_code == 200:
                    test_message = "One Call 3.0 OK"
                else:
                    try:
                        test_message = response.json().get("message", response.text[:200])
                    except Exception:
                        test_message = response.text[:200]
            except Exception as exc:
                test_status = 0
                test_message = str(exc)

        return {
            "env_file": str(_ENV_FILE),
            "env_file_exists": _ENV_FILE.is_file(),
            "key_length": len(runtime_key),
            "key_prefix": runtime_key[:4] if len(runtime_key) >= 4 else "",
            "key_suffix": runtime_key[-4:] if len(runtime_key) >= 4 else "",
            "file_key_matches_runtime": file_key == runtime_key,
            "file_key_hash": file_hash,
            "runtime_key_hash": runtime_hash,
            "onecall_3_0_test_status": test_status,
            "onecall_3_0_test_message": test_message,
        }
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get current weather conditions for a location
        Returns: dict with temperature, humidity, pressure, precipitation, etc.
        """
        try:
            data = self._fetch_onecall(lat, lon, exclude="minutely,hourly,daily,alerts")
            current = data.get('current')
            if not current:
                return self._get_fallback_current_weather()
            weather = (current.get('weather') or [{}])[0]
            
            return {
                'temperature': current.get('temp'),
                'feels_like': current.get('feels_like'),
                'humidity': current.get('humidity'),
                'pressure': current.get('pressure'),
                'precipitation': current.get('rain', {}).get('1h', 0) or current.get('snow', {}).get('1h', 0),
                'wind_speed': current.get('wind_speed', 0),
                'wind_direction': current.get('wind_deg', 0),
                'clouds': current.get('clouds', 0),
                'description': weather.get('description', 'N/A'),
                'icon': weather.get('icon'),
                'timestamp': datetime.now().isoformat(),
                'coordinates': {'lat': lat, 'lon': lon}
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            return self._get_fallback_current_weather()
        except Exception as e:
            print(f"Error processing current weather: {e}")
            return self._get_fallback_current_weather()
    
    def get_forecast(self, lat: float, lon: float, days: int = 5) -> Dict:
        """
        Get weather forecast for next N days using One Call 3.0 daily data.
        """
        try:
            data = self._fetch_onecall(lat, lon, exclude="minutely,hourly,alerts")
            daily = data.get('daily', [])
            
            forecast_list = []
            for day in daily[:days]:
                date_key = datetime.fromtimestamp(day.get('dt', datetime.utcnow().timestamp())).date().isoformat()
                temps = day.get('temp', {})
                humidity = day.get('humidity', 0)
                pressure = day.get('pressure', 0)
                wind_speed = day.get('wind_speed', 0)
                wind_gust = day.get('wind_gust', wind_speed)
                precipitation_total = day.get('rain') or day.get('snow') or 0.0
                weather = (day.get('weather') or [{}])[0]
                
                forecast_list.append({
                    'date': date_key,
                    'temperature': {
                        'min': temps.get('min'),
                        'max': temps.get('max'),
                        'avg': temps.get('day')
                    },
                    'precipitation': {
                        'total': precipitation_total,
                        'max_3h': precipitation_total
                    },
                    'humidity': {
                        'min': humidity,
                        'max': humidity,
                        'avg': humidity
                    },
                    'pressure': {
                        'min': pressure,
                        'max': pressure,
                        'avg': pressure
                    },
                    'wind_speed': {
                        'avg': wind_speed,
                        'max': max(wind_speed, wind_gust or 0)
                    },
                    'description': weather.get('description', 'unknown')
                })
            
            return {
                'location': {'lat': lat, 'lon': lon},
                'forecast_days': len(forecast_list),
                'forecasts': forecast_list,
                'timestamp': datetime.now().isoformat(),
                'source': 'OpenWeatherMap One Call 3.0'
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast: {e}")
            return self._get_fallback_forecast(days)
        except Exception as e:
            print(f"Error processing forecast: {e}")
            return self._get_fallback_forecast(days)
    
    def get_forecast_for_lstm(self, lat: float, lon: float, days: int = 7) -> List[Dict]:
        """
        Get forecast formatted for LSTM model input
        Returns list of daily forecasts with features needed for prediction
        """
        forecast_data = self.get_forecast(lat, lon, days)
        
        lstm_features = []
        for day_forecast in forecast_data.get('forecasts', []):
            lstm_features.append({
                'date': day_forecast['date'],
                'precipitation': day_forecast['precipitation']['total'],
                'temperature': day_forecast['temperature']['avg'],
                'humidity': day_forecast['humidity']['avg'],
                'pressure': day_forecast['pressure']['avg'],
                'wind_speed': day_forecast['wind_speed']['avg']
            })
        
        return lstm_features
    
    def get_air_quality(self, lat: float, lon: float) -> Dict:
        """Qualité de l'air (OpenWeather Air Pollution API)."""
        labels = {
            1: "Bonne",
            2: "Correcte",
            3: "Modérée",
            4: "Mauvaise",
            5: "Très mauvaise",
        }
        if not self.api_key:
            return {"aqi": None, "label": "Non configuré", "source": "fallback"}
        try:
            response = self._request_openweather(
                self.air_pollution_url, {"lat": lat, "lon": lon}
            )
            if response.status_code != 200:
                self._log_openweather_error("air_pollution", response)
                return {
                    "aqi": None,
                    "label": "Indisponible",
                    "source": "fallback",
                    "error": f"HTTP {response.status_code}",
                }
            payload = response.json()
            entry = (payload.get("list") or [{}])[0]
            aqi = int((entry.get("main") or {}).get("aqi", 2))
            aqi = max(1, min(5, aqi))
            return {
                "aqi": aqi,
                "label": labels.get(aqi, "—"),
                "components": entry.get("components"),
                "source": "openweather_air_pollution",
            }
        except Exception as exc:
            print(f"Error fetching air quality: {exc}")
            return {"aqi": None, "label": "Indisponible", "source": "fallback", "error": str(exc)}

    def get_weather_snapshot(self, lat: float, lon: float) -> Dict:
        """
        Menu radial : One Call API 3.0 uniquement (abonnement « by call »).
        Un seul appel (current + hourly). Cache 10 min. Pas d'API 2.5/weather.
        """
        cache_key = f"{round(lat, 3)}:{round(lon, 3)}"
        cached = _SNAPSHOT_CACHE.get(cache_key)
        if cached and datetime.now() - cached[0] < _SNAPSHOT_CACHE_TTL:
            return dict(cached[1])

        snapshot, err = self._weather_snapshot_from_onecall_v3(lat, lon)
        if snapshot is not None:
            _SNAPSHOT_CACHE[cache_key] = (datetime.now(), dict(snapshot))
            return snapshot

        if cached:
            stale = dict(cached[1])
            stale["stale"] = True
            stale["weather_error"] = err
            return stale

        return {
            "success": False,
            "coordinates": {"lat": lat, "lon": lon},
            "weather_error": err or "One Call 3.0 indisponible",
            "source": "openweather_onecall_3.0",
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_total_precipitation(self, lat: float, lon: float, days: int = 5) -> Dict:
        """
        Get total precipitation (rain + snow) for the next N days for a location.
        Returns dict containing total precipitation in millimeters and daily breakdown.
        """
        forecast_data = self.get_forecast(lat, lon, days)
        forecasts = forecast_data.get('forecasts', [])
        
        daily_breakdown = []
        total_precip = 0.0
        
        for day in forecasts:
            day_total = day.get('precipitation', {}).get('total', 0) or 0
            daily_breakdown.append({
                'date': day.get('date'),
                'total_mm': day_total
            })
            total_precip += day_total
        
        return {
            'location': {'lat': lat, 'lon': lon},
            'days': len(daily_breakdown),
            'total_precipitation_mm': total_precip,
            'daily_breakdown': daily_breakdown,
            'source': forecast_data.get('source', 'OpenWeatherMap One Call 3.0'),
            'timestamp': datetime.now().isoformat()
        }

    def _request_openweather(self, url: str, params: Dict) -> requests.Response:
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured")
        full_params = {**params, "appid": self.api_key, "units": "metric", "lang": "fr"}
        return requests.get(url, params=full_params, timeout=10)

    def _log_openweather_error(self, label: str, response: requests.Response) -> None:
        try:
            body = response.json()
            msg = body.get("message", body)
        except Exception:
            msg = response.text[:200]
        print(f"OpenWeather [{label}] HTTP {response.status_code}: {msg}")

    def _fetch_onecall_v3(
        self, lat: float, lon: float, exclude: Optional[str] = None
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """One Call API 3.0 — retourne (data, erreur) sans lever d'exception."""
        params = {"lat": lat, "lon": lon}
        if exclude:
            params["exclude"] = exclude
        try:
            response = self._request_openweather(self.one_call_url_v3, params)
        except Exception as exc:
            return None, str(exc)
        if response.status_code == 200:
            return response.json(), None
        self._log_openweather_error("onecall_3.0", response)
        import hashlib
        key_hint = hashlib.sha256(self.api_key.encode()).hexdigest()[:8]
        print(f"   ↳ clé utilisée (sha256[:8]): {key_hint}")
        try:
            body = response.json()
            msg = body.get("message", f"HTTP {response.status_code}")
        except Exception:
            msg = f"HTTP {response.status_code}"
        return None, msg

    def _fetch_onecall(self, lat: float, lon: float, exclude: Optional[str] = None) -> Dict:
        """Alias LSTM / prévisions — One Call 3.0 (lève si échec)."""
        data, err = self._fetch_onecall_v3(lat, lon, exclude=exclude)
        if data is None:
            raise requests.exceptions.HTTPError(err or "One Call 3.0 failed")
        return data

    @staticmethod
    def _precip_from_current(current: Dict) -> float:
        rain = current.get("rain") or {}
        if isinstance(rain, dict):
            return float(rain.get("1h") or 0.0)
        snow = current.get("snow") or {}
        if isinstance(snow, dict):
            return float(snow.get("1h") or 0.0)
        return 0.0

    def _weather_snapshot_from_onecall_v3(
        self, lat: float, lon: float
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Température, pluie 24h et humidité via un seul appel One Call 3.0."""
        data, err = self._fetch_onecall_v3(
            lat, lon, exclude="minutely,daily,alerts"
        )
        if data is None:
            return None, err

        current = data.get("current") or {}
        hourly = data.get("hourly") or []

        precip_1h = self._precip_from_current(current)
        precip_24h = 0.0
        for hour in hourly[:24]:
            rain_h = hour.get("rain") or {}
            if isinstance(rain_h, dict) and rain_h.get("1h") is not None:
                precip_24h += float(rain_h["1h"])

        weather_list = current.get("weather") or []
        desc = "—"
        if weather_list:
            desc = weather_list[0].get("description", desc)

        air = self.get_air_quality(lat, lon)

        return {
            "success": True,
            "coordinates": {"lat": lat, "lon": lon},
            "temperature_c": current.get("temp"),
            "humidity_pct": current.get("humidity"),
            "precipitation_1h_mm": precip_1h,
            "precipitation_24h_mm": round(precip_24h, 2),
            "description": desc,
            "air_quality_index": air.get("aqi"),
            "air_quality_label": air.get("label"),
            "timestamp": datetime.now().isoformat(),
            "source": "openweather_onecall_3.0",
            "stale": False,
        }, None

    def _fetch_current_weather_v25(self, lat: float, lon: float) -> Dict:
        """Météo actuelle via API 2.5 (incluse dans l'offre gratuite)."""
        response = self._request_openweather(
            self.current_weather_url, {"lat": lat, "lon": lon}
        )
        if response.status_code != 200:
            self._log_openweather_error("weather_2.5", response)
        response.raise_for_status()
        payload = response.json()
        weather = (payload.get("weather") or [{}])[0]
        rain = payload.get("rain") or {}
        snow = payload.get("snow") or {}
        precip = 0.0
        if isinstance(rain, dict):
            precip = float(rain.get("1h") or rain.get("3h") or 0.0)
        elif isinstance(snow, dict):
            precip = float(snow.get("1h") or snow.get("3h") or 0.0)
        return {
            "temp": payload.get("main", {}).get("temp"),
            "humidity": payload.get("main", {}).get("humidity"),
            "pressure": payload.get("main", {}).get("pressure"),
            "rain": {"1h": precip} if precip else {},
            "weather": [{"description": weather.get("description", "—")}],
        }

    def _weather_snapshot_from_v25(self, lat: float, lon: float) -> Dict:
        """Menu radial : météo actuelle 2.5 + qualité de l'air (2 appels max)."""
        current = self._fetch_current_weather_v25(lat, lon)
        rain_obj = current.get("rain") or {}
        precip_1h = float(rain_obj.get("1h") or 0.0) if isinstance(rain_obj, dict) else 0.0
        air = self.get_air_quality(lat, lon)
        weather_list = current.get("weather") or []
        desc = weather_list[0].get("description", "—") if weather_list else "—"
        return {
            "success": True,
            "coordinates": {"lat": lat, "lon": lon},
            "temperature_c": current.get("temp"),
            "humidity_pct": current.get("humidity"),
            "precipitation_1h_mm": precip_1h,
            "precipitation_24h_mm": round(precip_1h * 3, 2),
            "description": desc,
            "air_quality_index": air.get("aqi"),
            "air_quality_label": air.get("label"),
            "timestamp": datetime.now().isoformat(),
            "source": "openweather_2.5_current",
            "note": "Pluie 24h estimée — One Call 3.0 utilisé si quota disponible",
        }
    
    def _get_fallback_current_weather(self) -> Dict:
        """Fallback current weather data when API is unavailable"""
        return {
            'temperature': 30.0,
            'feels_like': 32.0,
            'humidity': 60.0,
            'pressure': 1013.0,
            'precipitation': 0.0,
            'wind_speed': 5.0,
            'wind_direction': 0,
            'clouds': 30,
            'description': 'partiellement nuageux',
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
    
    def _get_fallback_forecast(self, days: int) -> Dict:
        """Fallback forecast data when API is unavailable"""
        forecasts = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).date().isoformat()
            forecasts.append({
                'date': date,
                'temperature': {'min': 25.0, 'max': 35.0, 'avg': 30.0},
                'precipitation': {'total': 0.0, 'max_3h': 0.0},
                'humidity': {'min': 50.0, 'max': 70.0, 'avg': 60.0},
                'pressure': {'min': 1010.0, 'max': 1015.0, 'avg': 1013.0},
                'wind_speed': {'avg': 5.0, 'max': 10.0},
                'description': 'conditions normales'
            })
        
        return {
            'forecast_days': days,
            'forecasts': forecasts,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }

