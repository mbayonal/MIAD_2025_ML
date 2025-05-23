# Importación librerías
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Songs Popularity Prediction API',
    description='API to predict song popularity using XGBoost model')

ns = api.namespace('predict', 
     description='Songs Popularity Classifier')

# Definición argumentos o parámetros de la API
parser = api.parser()

parser.add_argument(
    'acousticness', 
    type=float, 
    required=True, 
    help='Acousticness feature of the song', 
    location='args')
    
parser.add_argument(
    'danceability', 
    type=float, 
    required=True, 
    help='Danceability feature of the song', 
    location='args')
    
parser.add_argument(
    'energy', 
    type=float, 
    required=True, 
    help='Energy feature of the song', 
    location='args')
    
parser.add_argument(
    'instrumentalness', 
    type=float, 
    required=True, 
    help='Instrumentalness feature of the song', 
    location='args')
    
parser.add_argument(
    'liveness', 
    type=float, 
    required=True, 
    help='Liveness feature of the song', 
    location='args')
    
parser.add_argument(
    'loudness', 
    type=float, 
    required=True, 
    help='Loudness feature of the song', 
    location='args')
    
parser.add_argument(
    'speechiness', 
    type=float, 
    required=True, 
    help='Speechiness feature of the song', 
    location='args')
    
parser.add_argument(
    'tempo', 
    type=float, 
    required=True, 
    help='Tempo feature of the song', 
    location='args')
    
parser.add_argument(
    'valence', 
    type=float, 
    required=True, 
    help='Valence feature of the song', 
    location='args')

parser.add_argument(
    'duration_ms', 
    type=float, 
    required=True, 
    help='Duration in milliseconds', 
    location='args')
    
parser.add_argument(
    'genre_encoded', 
    type=int, 
    required=True, 
    help='Encoded genre value', 
    location='args')

model = joblib.load('songs_popularity.pkl')

resource_fields = api.model('Resource', {
    'popularity': fields.Float,
})

# Definición de la clase para disponibilización
@ns.route('/')
class SongsPopularityApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        # Parseo de features en arrays con el orden correcto que espera el modelo
        features = np.array([[
            args['duration_ms'],
            args['danceability'],
            args['energy'],
            args['loudness'],
            args['speechiness'],
            args['acousticness'],
            args['instrumentalness'],
            args['liveness'],
            args['valence'],
            args['tempo'],
            args['genre_encoded']
        ]])
        
        #asigna el nombre de los predictores
        feature_names = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 
                         'tempo', 'genre_encoded']
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Para un modelo de regresión XGBRegressor
        prediction = model.predict(features_df)[0]
        
        return {
            "popularity": float(prediction)  # Convertir a float para serialización JSON
        }, 200

if __name__ == '__main__':
    # Configuración para entorno de producción
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=port)