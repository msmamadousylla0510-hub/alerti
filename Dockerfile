# API Alerti (Flask + TensorFlow) — build explicite pour Railway (évite Railpack vide).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dépendances système (TensorFlow CPU, OpenCV headless, rasterio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV TF_ENABLE_ONEDNN_OPTS=0

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY app.py .
COPY backend ./backend

# Modèles Bamako (doivent être versionnés — voir RAILWAY_DEPLOY.md)
# COPY échoue au build si les fichiers manquent dans le contexte Git.

EXPOSE 8080

# Railway injecte $PORT
CMD ["sh", "-c", "python app.py"]
