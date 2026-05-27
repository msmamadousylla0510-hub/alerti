# Déploiement Railway — Alerti (Python)

## Erreur « zero build logs / Railpack ne démarre pas »

Souvent causé par :

1. **Aucun builder explicite** (corrigé : `railway.json` + `Dockerfile`)
2. **Mauvais répertoire racine** dans Railway → Settings → Root Directory = `/` (racine du repo `alerti`)
3. **Modèles `.h5` non versionnés** (gitignore) → image sans modèle
4. Bug plateforme → redeploy après push de `Dockerfile`

## Étapes

### 1. Versionner le modèle Bamako (obligatoire)

Les fichiers sont ignorés par défaut. Les inclure pour la prod :

```bash
cd alerti
git add -f backend/models/lstm_model_bamako.h5
git add -f backend/models/lstm_scaler_bamako.pkl
git commit -m "Add Bamako LSTM artifacts for Railway deploy"
git push
```

(~5 Mo total)

### 2. Variables Railway (Settings → Variables)

Copier depuis `.env` :

- `OPENWEATHERMAP_API_KEY` (**obligatoire** pour `/api/weather/at`)
- `PORT` est défini automatiquement par Railway
- Optionnel : Twilio, FCM, etc.

Ne pas uploader `.env` tel quel si le repo est public.

Après deploy, vérifier :

```bash
curl https://VOTRE-SERVICE.up.railway.app/api/health
```

Si `startup_errors` contient `bamako` ou `weather`, lire le message (souvent clé API ou modèle `.h5` manquant).

### 3. Forcer le builder Docker

Dans Railway → Service → **Settings** :

- **Builder** : `Dockerfile`
- **Dockerfile path** : `Dockerfile`
- **Root Directory** : vide ou `.` (racine du repo alerti)

Ou laisser `railway.json` à la racine.

### 4. Déployer

```bash
git push origin main
```

Build attendu : 10–20 min (TensorFlow). Des logs Docker doivent apparaître.

### 5. Tester

```bash
curl https://VOTRE-SERVICE.up.railway.app/
curl "https://VOTRE-SERVICE.up.railway.app/api/weather/at?lat=12.65&lon=-7.98"
curl -X POST https://VOTRE-SERVICE.up.railway.app/api/bamako/predict \
  -H "Content-Type: application/json" \
  -d '{"neighborhood":"Sebenikoro"}'
```

### 6. Flutter

```bash
flutter run --dart-define=ALERTI_API_BASE=https://VOTRE-SERVICE.up.railway.app
```

## Si le build échoue encore

| Symptôme | Action |
|----------|--------|
| Toujours 0 log | Nouveau service Railway, reconnecter le repo GitHub |
| `COPY` / modèle manquant | `git add -f` des `.h5` / `.pkl` Bamako |
| OOM / timeout pip | Plan Railway avec plus de RAM ; ou alléger `requirements.txt` |
| Railpack au lieu de Docker | Désactiver Railpack : builder = **Dockerfile** dans l’UI |

## Vercel

Ne pas déployer `alerti` sur Vercel (TensorFlow > 500 Mo). Railway uniquement pour cette API.
