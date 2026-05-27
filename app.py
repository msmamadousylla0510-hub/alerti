"""
Flood Forecast Application - Main Flask App
Predicts floods for African countries using hybrid AI model (LSTM + CNN)
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Optional

# Add backend directory to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from utils.config import AFRICAN_LOCATIONS, MALI_CITIES

_startup_errors: dict[str, str] = {}
_hybrid_predictor = None
_hybrid_load_error: Optional[str] = None


def _load_service(label: str, factory):
    """Charge un service isolément (un échec n'empêche pas les autres)."""
    try:
        service = factory()
        print(f"[startup] OK {label}")
        return service
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        _startup_errors[label] = msg
        print(f"[startup] FAIL {label}: {msg}")
        traceback.print_exc()
        return None


def _create_weather_service():
    from services.weather_forecast_service import WeatherForecastService
    return WeatherForecastService()


def _create_bamako_service():
    from services.bamako_prediction_service import BamakoPredictionService
    return BamakoPredictionService()


def _create_neighborhood_service():
    from services.neighborhood_service import NeighborhoodService
    return NeighborhoodService()


def _create_africa_data_service():
    from services.africa_data_service import AfricaDataService
    return AfricaDataService()


def _create_alert_service():
    from services.alert_service import AlertService
    return AlertService()


def _create_notification_service():
    from services.notification_service import NotificationService
    return NotificationService()


def _create_push_service():
    from services.push_notification_service import PushNotificationService
    return PushNotificationService()


def _create_satellite_service():
    from services.satellite_service import SatelliteService
    return SatelliteService()


def get_hybrid_predictor():
    """Charge le modèle hybride LSTM+CNN à la demande (évite de bloquer le démarrage)."""
    global _hybrid_predictor, _hybrid_load_error
    if _hybrid_predictor is not None:
        return _hybrid_predictor
    if _hybrid_load_error:
        raise RuntimeError(_hybrid_load_error)
    try:
        from models.hybrid_model import HybridFloodPredictor
        _hybrid_predictor = HybridFloodPredictor()
        print("[startup] OK hybrid (lazy)")
        return _hybrid_predictor
    except Exception as exc:
        _hybrid_load_error = f"{type(exc).__name__}: {exc}"
        _startup_errors["hybrid"] = _hybrid_load_error
        traceback.print_exc()
        raise RuntimeError(_hybrid_load_error) from exc


app = Flask(__name__)
CORS(app)

weather_forecast_service = _load_service("weather", _create_weather_service)
bamako_prediction_service = _load_service("bamako", _create_bamako_service)
neighborhood_service = _load_service("neighborhood", _create_neighborhood_service)
africa_data_service = _load_service("africa_data", _create_africa_data_service)
alert_service = _load_service("alert", _create_alert_service)
notification_service = _load_service("notification", _create_notification_service)
push_notification_service = _load_service("push", _create_push_service)
satellite_service = _load_service("satellite", _create_satellite_service)

if weather_forecast_service:
    _diag = weather_forecast_service.key_diagnostics()
    _ow_len = _diag.get("key_length", 0)
    if _ow_len:
        _pfx = _diag.get("key_prefix", "")
        _sfx = _diag.get("key_suffix", "")
        _match = _diag.get("file_key_matches_runtime")
        _st = _diag.get("onecall_3_0_test_status")
        print(
            f"🌤️ OpenWeather: clé {_pfx}…{_sfx} ({_ow_len} car.) "
            f"test 3.0 HTTP {_st}"
        )
        if _st != 200:
            print(f"   ↳ {_diag.get('onecall_3_0_test_message', '')[:200]}")
    else:
        print("⚠️ OpenWeather: OPENWEATHERMAP_API_KEY manquante (variables Railway)")


def _predict_neighborhood_core(neighborhood_name, city, bbox=None):
    """
    Prédiction quartier : Bamako → modèle LSTM Bamako (comme le dashboard web),
    autres villes → hybride LSTM+CNN.
    """
    city_lower = city.lower()
    coords = neighborhood_service.get_neighborhood_coordinates(
        neighborhood_name, city_lower
    )
    if not coords:
        return None, {'error': f'Neighborhood {neighborhood_name} not found in {city}'}, 404

    lat = coords['lat']
    lon = coords['lon']
    if bbox is None:
        bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]

    forecast_data = None
    if weather_forecast_service:
        forecast_data = weather_forecast_service.get_forecast_for_lstm(lat, lon, days=7)

    prediction = None
    model_source = 'hybrid'
    commune = None
    context = None
    metadata = None

    if city_lower == 'bamako' and bamako_prediction_service:
        try:
            bamako_result = bamako_prediction_service.predict(
                neighborhood=neighborhood_name
            )
            prediction = dict(bamako_result.get('prediction') or {})
            prediction['coordinates'] = coords
            model_source = prediction.get('source', 'lstm_bamako')
            commune = bamako_result.get('commune')
            context = bamako_result.get('context')
            metadata = bamako_result.get('metadata')
        except (ValueError, Exception) as exc:
            app.logger.warning(
                "Bamako LSTM indisponible pour %s, repli hybride: %s",
                neighborhood_name,
                exc,
            )

    if prediction is None:
        prediction = get_hybrid_predictor().predict(
            lat,
            lon,
            bbox,
            f"{neighborhood_name}, {city}",
            forecast_data=forecast_data,
        )
        prediction = dict(prediction)
        prediction['coordinates'] = coords
        model_source = prediction.get('model_type', 'hybrid')

    risk_level = prediction.get('risk_level', 'none')
    payload = {
        'neighborhood': neighborhood_name,
        'city': city,
        'coordinates': coords,
        'prediction': prediction,
        'model_source': model_source,
        'forecast': forecast_data,
        'recommendations': alert_service.get_recommendations(risk_level),
        'timestamp': datetime.now().isoformat(),
    }
    if commune:
        payload['commune'] = commune
    if context:
        payload['context'] = context
    if metadata:
        payload['metadata'] = metadata

    alert = None
    if prediction.get('flood_probability', 0) > 0.2:
        alert = alert_service.create_alert(
            f"{neighborhood_name}, {city}",
            risk_level,
            prediction,
            lat=lat,
            lon=lon,
        )
        if risk_level in ['critical', 'high']:
            notification_service.send_alert_notifications(alert)
            if push_notification_service:
                push_notification_service.send_alert_to_location(
                    alert,
                    location=city,
                    neighborhood=neighborhood_name,
                )
    payload['alert'] = alert
    return payload, None, 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Diagnostic démarrage (Railway / debug)."""
    return jsonify({
        'status': 'ok' if not _startup_errors else 'degraded',
        'services': {
            'weather': weather_forecast_service is not None,
            'bamako': bamako_prediction_service is not None,
            'neighborhood': neighborhood_service is not None,
            'hybrid_lazy': _hybrid_predictor is not None,
        },
        'startup_errors': _startup_errors,
        'openweather': (
            weather_forecast_service.key_diagnostics()
            if weather_forecast_service
            else None
        ),
    })


@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Flood Forecast API for Africa',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'predict': '/api/predict',
            'predict_meteo': '/api/predict-meteo',
            'predict_image': '/api/predict-image',
            'bamako_predict': '/api/bamako/predict',
            'predict_neighborhood': '/api/predict/neighborhood',
            'mali_neighborhoods': '/api/mali/neighborhoods',
            'mali_neighborhood_predict': '/api/mali/neighborhood/<name>/predict',
            'alerts': '/api/alerts',
            'forecast': '/api/forecast/<country>',
            'subscribe': '/api/alert/subscribe',
            'subscribe_push': '/api/subscribe/push',
            'satellite_image': '/api/satellite-image/<location>',
            'countries': '/api/countries',
            'weather_at': '/api/weather/at?lat=&lon=',
        }
    })


@app.route('/api/weather/diag', methods=['GET'])
def weather_key_diagnostics():
    """Vérifie quelle clé OpenWeather est utilisée (empreinte, pas la clé complète)."""
    if not weather_forecast_service:
        return jsonify({
            'error': 'Weather service not available',
            'startup_errors': _startup_errors,
        }), 503
    return jsonify(weather_forecast_service.key_diagnostics())


@app.route('/api/weather/at', methods=['GET', 'OPTIONS'])
def weather_at_coordinates():
    """Météo + pluie 24h + qualité de l'air pour des coordonnées GPS."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if not weather_forecast_service:
        return jsonify({
            'error': 'Weather service not available',
            'startup_errors': _startup_errors,
        }), 503
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None:
        return jsonify({'error': 'Query params lat and lon required'}), 400
    snapshot = weather_forecast_service.get_weather_snapshot(lat, lon)
    status = 200 if snapshot.get("success", True) else 503
    return jsonify(snapshot), status


@app.route('/api/predict', methods=['POST'])
def predict_flood():
    """Complete flood prediction combining meteo and satellite data"""
    try:
        data = request.json
        
        location_name = data.get('location', 'Unknown')
        lat = data.get('latitude')
        lon = data.get('longitude')
        country = data.get('country', '').lower()
        
        if not lat or not lon:
            # Try to get from country name
            if country and country in AFRICAN_LOCATIONS:
                loc_data = AFRICAN_LOCATIONS[country]
                lat = loc_data['lat']
                lon = loc_data['lon']
                location_name = loc_data['name']
            else:
                return jsonify({'error': 'Latitude and longitude required'}), 400
        
        bbox = data.get('bbox', [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1])
        
        # Get comprehensive prediction using hybrid model
        prediction = get_hybrid_predictor().predict(lat, lon, bbox, location_name)
        
        # Create alert if needed
        if prediction.get('flood_probability', 0) > 0.2:
            alert = alert_service.create_alert(
                location_name,
                prediction.get('risk_level', 'low'),
                prediction,
                lat=lat,
                lon=lon
            )
            
            # Send notifications for high-level alerts
            if prediction.get('risk_level') in ['critical', 'high']:
                notification_service.send_alert_notifications(alert)
        else:
            alert = None
        
        return jsonify({
            'location': location_name,
            'coordinates': {'lat': lat, 'lon': lon},
            'prediction': prediction,
            'alert': alert,
            'recommendations': alert_service.get_recommendations(prediction.get('risk_level', 'none')),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-meteo', methods=['POST'])
def predict_flood_meteo():
    """Flood prediction based on meteorological data only"""
    try:
        data = request.json
        
        lat = data.get('latitude')
        lon = data.get('longitude')
        location_name = data.get('location', 'Unknown')
        
        if not lat or not lon:
            return jsonify({'error': 'Latitude and longitude required'}), 400
        
        # Use LSTM model only
        _hybrid = get_hybrid_predictor()
        if hasattr(_hybrid, 'lstm_predictor'):
            prediction = _hybrid.lstm_predictor.predict(lat, lon, location_name)
        else:
            prediction = {'flood_probability': 0.3, 'risk_level': 'low', 'model_type': 'LSTM'}
        
        return jsonify({
            'location': location_name,
            'coordinates': {'lat': lat, 'lon': lon},
            'prediction': prediction,
            'model_type': 'LSTM (Meteorological)',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bamako/predict', methods=['POST', 'OPTIONS'])
def predict_bamako_commune():
    """Expose the Bamako-specific LSTM predictions (commune/quartier)."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not bamako_prediction_service:
        return jsonify({
            'error': 'Bamako prediction service not available',
            'startup_errors': _startup_errors,
        }), 503

    data = request.json or {}
    commune = data.get('commune')
    neighborhood = data.get('neighborhood')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if neighborhood and not commune:
        from utils.bamako_neighborhoods import (
            get_commune_from_neighborhood,
            resolve_neighborhood_from_localite,
        )
        if not get_commune_from_neighborhood(neighborhood):
            resolved = resolve_neighborhood_from_localite(neighborhood)
            if resolved:
                neighborhood = resolved

    if not commune and not neighborhood:
        return jsonify({'error': 'Provide either a commune or a neighborhood name'}), 400

    try:
        result = bamako_prediction_service.predict(
            commune=commune,
            neighborhood=neighborhood,
            latitude=latitude,
            longitude=longitude,
        )
        if neighborhood and not result.get('neighborhood'):
            result['neighborhood'] = neighborhood

        prediction = result.get('prediction', {})
        app.logger.info(
            "[BamakoPredict] commune=%s neighborhood=%s risk=%s prob=%.3f mode=%s end=%s latency=%sms",
            result.get('commune'),
            result.get('neighborhood'),
            prediction.get('risk_level'),
            prediction.get('flood_probability', 0.0),
            result.get('metadata', {}).get('inference_mode'),
            prediction.get('sequence_end_date'),
            result.get('metadata', {}).get('latency_ms'),
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Bamako prediction failed: %s", exc)
        return jsonify({'error': 'Internal error while running Bamako prediction'}), 500

@app.route('/api/predict-image', methods=['POST'])
def predict_flood_image():
    """Flood prediction based on satellite image only"""
    try:
        data = request.json
        
        lat = data.get('latitude')
        lon = data.get('longitude')
        location_name = data.get('location', 'Unknown')
        bbox = data.get('bbox', [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1])
        
        if not lat or not lon:
            return jsonify({'error': 'Latitude and longitude required'}), 400
        
        # Use CNN model only
        _hybrid = get_hybrid_predictor()
        if hasattr(_hybrid, 'cnn_predictor'):
            prediction = _hybrid.cnn_predictor.predict_image(lat, lon, bbox, location_name)
        else:
            prediction = {'flood_probability': 0.2, 'risk_level': 'low', 'model_type': 'CNN'}
        
        return jsonify({
            'location': location_name,
            'coordinates': {'lat': lat, 'lon': lon},
            'prediction': prediction,
            'model_type': 'CNN (Satellite Image)',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all active alerts"""
    try:
        country = request.args.get('country', '').lower()
        min_risk = request.args.get('min_risk', 'low')
        
        alerts = alert_service.get_active_alerts(country=country, min_risk=min_risk)
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/<country>', methods=['GET'])
def get_forecast(country):
    """Get flood forecast for a specific African country"""
    try:
        country_lower = country.lower()
        
        if country_lower not in AFRICAN_LOCATIONS:
            return jsonify({
                'error': f'Country {country} not found. Available countries: {list(AFRICAN_LOCATIONS.keys())}'
            }), 404
        
        loc_data = AFRICAN_LOCATIONS[country_lower]
        lat = loc_data['lat']
        lon = loc_data['lon']
        location_name = loc_data['name']
        bbox = [lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5]
        
        # Get comprehensive prediction
        prediction = get_hybrid_predictor().predict(lat, lon, bbox, location_name)
        
        # Get historical data for context
        historical_data = africa_data_service.get_chirps_precipitation(
            lat, lon, 
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        return jsonify({
            'country': location_name,
            'coordinates': {'lat': lat, 'lon': lon},
            'forecast': prediction,
            'historical_context': {
                'precipitation_30d': historical_data.get('total_precipitation', 0),
                'average_daily': historical_data.get('average_daily_precipitation', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/satellite-image/<location>', methods=['GET'])
def get_satellite_image(location):
    """Get satellite image for location"""
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        bbox = request.args.get('bbox', type=str)
        
        if not lat or not lon:
            # Try to get from location name
            if location.lower() in AFRICAN_LOCATIONS:
                loc_data = AFRICAN_LOCATIONS[location.lower()]
                lat = loc_data['lat']
                lon = loc_data['lon']
            else:
                return jsonify({'error': 'Latitude and longitude required'}), 400
        
        if bbox:
            bbox = [float(x) for x in bbox.split(',')]
        else:
            bbox = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]
        
        if satellite_service:
            satellite_data = satellite_service.get_sentinel2_image(lat, lon, bbox)
        else:
            satellite_data = {'error': 'Satellite service not available'}
        
        return jsonify(satellite_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Get list of supported African countries"""
    countries = {
        code: {
            'name': data['name'],
            'coordinates': {'lat': data['lat'], 'lon': data['lon']}
        }
        for code, data in AFRICAN_LOCATIONS.items()
    }
    
    return jsonify({
        'countries': countries,
        'count': len(countries)
    })

@app.route('/api/mali/neighborhoods', methods=['GET'])
def get_mali_neighborhoods():
    """Get list of neighborhoods for Mali cities"""
    try:
        city = request.args.get('city', 'bamako').lower()
        
        if not neighborhood_service:
            return jsonify({'error': 'Neighborhood service not available'}), 503
        
        neighborhoods = neighborhood_service.get_mali_neighborhoods(city)
        
        return jsonify({
            'city': city,
            'neighborhoods': neighborhoods,
            'count': len(neighborhoods),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mali/neighborhood/<neighborhood_name>/predict', methods=['GET'])
def predict_neighborhood(neighborhood_name):
    """Get flood prediction for a specific neighborhood in Mali"""
    try:
        city = request.args.get('city', 'bamako').lower()
        
        if not neighborhood_service:
            return jsonify({'error': 'Neighborhood service not available'}), 503

        payload, err, status = _predict_neighborhood_core(neighborhood_name, city)
        if err:
            return jsonify(err), status
        return jsonify(payload)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/neighborhood', methods=['POST'])
def predict_neighborhood_post():
    """Predict flood for a neighborhood (POST method)"""
    try:
        data = request.json
        
        neighborhood_name = data.get('neighborhood')
        city = data.get('city', 'bamako').lower()
        
        if not neighborhood_name:
            return jsonify({'error': 'Neighborhood name required'}), 400
        
        if not neighborhood_service:
            return jsonify({'error': 'Neighborhood service not available'}), 503

        bbox = data.get('bbox')
        payload, err, status = _predict_neighborhood_core(
            neighborhood_name, city, bbox=bbox
        )
        if err:
            return jsonify(err), status
        return jsonify(payload)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribe/push', methods=['POST'])
def subscribe_push():
    """Subscribe to push notifications for a location/neighborhood"""
    try:
        data = request.json
        
        device_token = data.get('device_token')
        location = data.get('location')
        neighborhood = data.get('neighborhood')
        
        if not device_token:
            return jsonify({'error': 'Device token required'}), 400
        
        if not push_notification_service:
            return jsonify({'error': 'Push notification service not available'}), 503
        
        success = push_notification_service.register_device(
            device_token=device_token,
            location=location,
            neighborhood=neighborhood
        )
        
        return jsonify({
            'message': 'Successfully subscribed to push notifications',
            'device_token': device_token[:20] + '...',
            'location': location,
            'neighborhood': neighborhood,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alert/subscribe', methods=['POST'])
def subscribe_alerts():
    """Subscribe to flood alerts (updated to support WhatsApp and push)"""
    try:
        data = request.json
        
        email = data.get('email')
        phone = data.get('phone')
        device_token = data.get('device_token')
        location = data.get('location')
        neighborhood = data.get('neighborhood')
        country = data.get('country', '').lower()
        whatsapp = data.get('whatsapp', False)
        
        if not email and not phone and not device_token:
            return jsonify({'error': 'Email, phone, or device_token required'}), 400
        
        subscription = notification_service.subscribe(
            email=email,
            phone=phone,
            location=neighborhood or location,
            country=country
        )
        
        # Add WhatsApp preference
        if whatsapp and phone:
            subscription['whatsapp'] = True
        
        # Register for push notifications if device token provided
        push_subscription = None
        if device_token and push_notification_service:
            push_notification_service.register_device(
                device_token=device_token,
                location=location,
                neighborhood=neighborhood
            )
            push_subscription = {'registered': True}
        
        return jsonify({
            'message': 'Successfully subscribed to alerts',
            'subscription': subscription,
            'push_subscription': push_subscription,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, port=port, host='0.0.0.0')
