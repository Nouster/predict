from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import pickle

# Charger le modèle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Enums pour Swagger pour les groupes de quartiers et types de chambres
# Ces enums sont utilisés pour la validation des entrées et la documentation de l'API
class NeighbourhoodGroup(str, Enum):
    brooklyn = "Brooklyn"
    manhattan = "Manhattan"
    queens = "Queens"
    staten_island = "Staten Island"

class RoomType(str, Enum):
    private_room = "Private room"
    shared_room = "Shared room"
    entire_home_apt = "Entire home/apt"

# Schéma de requête
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    calculated_host_listings_count: int
    availability_365: int
    neighbourhood_group: NeighbourhoodGroup
    room_type: RoomType

app = FastAPI(title="API de prédiction Airbnb")

@app.post("/predict")
def predict_price(req: PredictionRequest):
    # Initialiser les champs
    input_data = {
        'latitude': req.latitude,
        'longitude': req.longitude,
        'minimum_nights': req.minimum_nights,
        'number_of_reviews': req.number_of_reviews,
        'reviews_per_month': req.reviews_per_month,
        'calculated_host_listings_count': req.calculated_host_listings_count,
        'availability_365': req.availability_365,
        'neighbourhood_group_Brooklyn': 0,
        'neighbourhood_group_Manhattan': 0,
        'neighbourhood_group_Queens': 0,
        'neighbourhood_group_Staten Island': 0,
        'room_type_Private room': 0,
        'room_type_Shared room': 0
    }

    # Encodage one-hot
    ng_key = f'neighbourhood_group_{req.neighbourhood_group.value}'
    if ng_key in input_data:
        input_data[ng_key] = 1

    if req.room_type == RoomType.private_room:
        input_data['room_type_Private room'] = 1
    elif req.room_type == RoomType.shared_room:
        input_data['room_type_Shared room'] = 1
    # "Entire home/apt" -> rien à changer (valeur implicite)

    # Création DataFrame
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

    # Prédiction
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(prediction, 2)}
