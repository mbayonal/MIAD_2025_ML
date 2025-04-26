# Songs Popularity Prediction API

API REST desarrollada en Flask para predecir la popularidad de canciones utilizando un modelo XGBoost.

## Requisitos

Consulta el archivo `requirements.txt` para ver las dependencias necesarias.

## Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

Para desarrollo:
```bash
python api.py
```

Para producción (con Gunicorn):
```bash
gunicorn --bind 0.0.0.0:5000 api:app
```

## Parámetros de la API

La API acepta los siguientes parámetros para predecir la popularidad de una canción:

- `acousticness`: Valor de acústica de la canción (float)
- `danceability`: Valor de bailabilidad (float)
- `energy`: Energía de la canción (float)
- `instrumentalness`: Valor instrumental (float)
- `liveness`: Valor de presencia de audiencia en vivo (float)
- `loudness`: Valor de volumen (float)
- `speechiness`: Valor de presencia de voz hablada (float)
- `tempo`: Tempo de la canción (float)
- `valence`: Valencia o positividad (float)

## Ejemplo de uso

```
GET /predict/?acousticness=0.1&danceability=0.8&energy=0.7&instrumentalness=0.01&liveness=0.15&loudness=-5.0&speechiness=0.05&tempo=120.0&valence=0.6
```

Respuesta:
```json
{
  "popularity": "Popular",
  "probability": 0.85
}
```
