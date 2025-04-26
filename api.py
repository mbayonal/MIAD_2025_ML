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

model = joblib.load('songs_popularity.pkl')

resource_fields = api.model('Resource', {
    'popularity': fields.String,
    'probability': fields.Float,
})

# Definición de la clase para disponibilización
@ns.route('/')
class SongsPopularityApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        # Parseo de features en arrays
        features = np.array([[
            args['acousticness'],
            args['danceability'],
            args['energy'],
            args['instrumentalness'],
            args['liveness'],
            args['loudness'],
            args['speechiness'],
            args['tempo'],
            args['valence']
        ]])
        
        #asigna el nombre de los predictores
        feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                        'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
        features_df = pd.DataFrame(features, columns=feature_names)
        
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]  # Probability of positive class
        
        result = "Popular" if prediction == 1 else "Not Popular"
        
        return {
            "popularity": result,
            "probability": float(probability)
        }, 200

if __name__ == '__main__':
    # Configuración para entorno de producción
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=port)