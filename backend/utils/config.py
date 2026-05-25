"""
Configuration file for Flood Forecast System
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Toujours charger alerti/.env (pas une variable shell obsolète ni un autre cwd)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE, override=True)

# API Keys (set in .env file)
EARTHDATA_USERNAME = os.getenv('EARTHDATA_USERNAME', '')
EARTHDATA_PASSWORD = os.getenv('EARTHDATA_PASSWORD', '')
COPERNICUS_USERNAME = os.getenv('COPERNICUS_USERNAME', '')
COPERNICUS_PASSWORD = os.getenv('COPERNICUS_PASSWORD', '')
SENTINEL_HUB_API_KEY = os.getenv('SENTINEL_HUB_API_KEY', '')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', '')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_USER = os.getenv('EMAIL_USER', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
OPENWEATHERMAP_API_KEY = (os.getenv('OPENWEATHERMAP_API_KEY') or '').strip()
FCM_SERVER_KEY = os.getenv('FCM_SERVER_KEY', '')
FCM_CREDENTIALS_PATH = os.getenv('FCM_CREDENTIALS_PATH', '')

# Data Sources URLs
CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"
GPM_BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov"
DE_AFRICA_BASE_URL = "https://explorer.digitalearth.africa"
COPERNICUS_URL = "https://apihub.copernicus.eu/apihub"
NASA_WORLDVIEW_URL = "https://wvs.earthdata.nasa.gov"

# Model Configuration
LSTM_SEQUENCE_LENGTH = 30  # Days of historical data
LSTM_FORECAST_DAYS = 7     # Days to predict ahead
CNN_INPUT_SIZE = (256, 256)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', 'models')

# Alert Thresholds
ALERT_THRESHOLDS = {
    'critical': 0.8,
    'high': 0.6,
    'medium': 0.4,
    'low': 0.2
}

# African Countries Coordinates (major cities)
AFRICAN_LOCATIONS = {
    'nigeria': {'lat': 9.0820, 'lon': 8.6753, 'name': 'Nigeria'},
    'kenya': {'lat': -0.0236, 'lon': 37.9062, 'name': 'Kenya'},
    'south_africa': {'lat': -25.7461, 'lon': 28.1881, 'name': 'South Africa'},
    'ghana': {'lat': 5.6037, 'lon': -0.1870, 'name': 'Ghana'},
    'tanzania': {'lat': -6.7924, 'lon': 39.2083, 'name': 'Tanzania'},
    'uganda': {'lat': 0.3476, 'lon': 32.5825, 'name': 'Uganda'},
    'mali': {'lat': 12.6392, 'lon': -8.0029, 'name': 'Mali'},
    'mozambique': {'lat': -25.9692, 'lon': 32.5732, 'name': 'Mozambique'},
    'cameroon': {'lat': 3.8480, 'lon': 11.5021, 'name': 'Cameroon'},
    'ivory_coast': {'lat': 5.3600, 'lon': -4.0083, 'name': "Côte d'Ivoire"},
    'madagascar': {'lat': -18.8792, 'lon': 47.5079, 'name': 'Madagascar'},
    'angola': {'lat': -8.8139, 'lon': 13.2319, 'name': 'Angola'},
    'niger': {'lat': 13.5137, 'lon': 2.1098, 'name': 'Niger'},
    'burkina_faso': {'lat': 12.2383, 'lon': -1.5616, 'name': 'Burkina Faso'},
    'malawi': {'lat': -13.9626, 'lon': 33.7741, 'name': 'Malawi'},
    'zambia': {'lat': -15.3875, 'lon': 28.3228, 'name': 'Zambia'},
    'senegal': {'lat': 14.7167, 'lon': -17.4677, 'name': 'Senegal'},
    'chad': {'lat': 12.1348, 'lon': 15.0557, 'name': 'Chad'},
    'somalia': {'lat': 2.0469, 'lon': 45.3182, 'name': 'Somalia'},
    'zimbabwe': {'lat': -17.8292, 'lon': 31.0522, 'name': 'Zimbabwe'}
}

# Mali Cities Coordinates
MALI_CITIES = {
    'bamako': {'lat': 12.6392, 'lon': -8.0029, 'name': 'Bamako'},
    'kayes': {'lat': 14.4469, 'lon': -11.4445, 'name': 'Kayes'},
    'sikasso': {'lat': 11.3167, 'lon': -5.6667, 'name': 'Sikasso'},
    'mopti': {'lat': 14.4869, 'lon': -4.2000, 'name': 'Mopti'},
    'segou': {'lat': 13.4400, 'lon': -6.2600, 'name': 'Ségou'},
    'tombouctou': {'lat': 16.7733, 'lon': -3.0074, 'name': 'Tombouctou'},
    'gao': {'lat': 16.2667, 'lon': -0.0500, 'name': 'Gao'},
    'kidal': {'lat': 18.4411, 'lon': 1.4078, 'name': 'Kidal'},
    'koulikoro': {'lat': 12.8667, 'lon': -7.5667, 'name': 'Koulikoro'},
    'koutiala': {'lat': 12.3833, 'lon': -5.4667, 'name': 'Koutiala'}
}

# Neighborhood cache file path
NEIGHBORHOOD_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'backend', 'data', 'mali_neighborhoods.json'
)

