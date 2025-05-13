import pytest
import pandas as pd
import pickle
import numpy as np
from pydantic import ValidationError
from main import PredictionRequest, NeighbourhoodGroup, RoomType


    # Fixture qui charge le modèle une seule fois pour tous les tests du module.
    # Ça nous évite de recharger le modèle à chaque test.
    # Le modèle est supposé avoir été sauvegardé dans un fichier 'model.pkl'.
    
@pytest.fixture(scope='module')
def model():
    
    return pickle.load(open('model.pkl', 'rb'))



# Fixture qui récupère les noms des caractéristiques attendus par mon modèle.
# Ça garantit que les données de test ont bien la bonne structure (même noms de colonnes).

@pytest.fixture(scope='module')
def feature_names(model):
    return model.feature_names_in_

def test_valid_input(model, feature_names):
    """Test avec des données valides, attend une prédiction correcte"""
    input_data = {
        'latitude': 40.7128,
        'longitude': -74.0060,
        'minimum_nights': 3,
        'number_of_reviews': 10,
        'reviews_per_month': 1.2,
        'calculated_host_listings_count': 2,
        'availability_365': 180,
        'neighbourhood_group_Brooklyn': 0,
        'neighbourhood_group_Manhattan': 1,
        'neighbourhood_group_Queens': 0,
        'neighbourhood_group_Staten Island': 0,
        'room_type_Private room': 0,
        'room_type_Shared room': 0
    }
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(input_df)
    assert isinstance(prediction, np.ndarray)  # Je vérifie bien que le retour est un numpy


# Test avec un dictionnaire contenant uniquement des champs vides (chaînes vides).

def test_empty_input(model, feature_names):
    """Test avec une entrée vide"""
    input_data = {feature: '' for feature in feature_names}
    input_df = pd.DataFrame([input_data], columns=feature_names)
    with pytest.raises(ValueError):  
        model.predict(input_df)

def test_special_characters_input(model, feature_names):
    """Test avec des caractères spéciaux dans les inputs"""
    input_data = {
        'latitude': '40.7128@',  # Caractère spécial
        'longitude': -74.060,
        'minimum_nights': 3,
        'number_of_reviews': 10,
        'reviews_per_month': 1.2,
        'calculated_host_listings_count': 2,
        'availability_365': 180,
        'neighbourhood_group_Brooklyn': 0,
        'neighbourhood_group_Manhattan': 0,
        'neighbourhood_group_Queens': 0,
        'neighbourhood_group_Staten Island': 0,
        'room_type_Private room': 0,
        'room_type_Shared room': 0
    }
    input_df = pd.DataFrame([input_data], columns=feature_names)
    with pytest.raises(ValueError):  # Vérifie si une erreur est levée pour les caractères invalides
        model.predict(input_df)

def test_valid_enums():
    # ✅ Cas valides
    req = PredictionRequest(
        latitude=40.7,
        longitude=-73.9,
        minimum_nights=2,
        number_of_reviews=10,
        reviews_per_month=1.2,
        calculated_host_listings_count=1,
        availability_365=100,
        neighbourhood_group=NeighbourhoodGroup.brooklyn,
        room_type=RoomType.private_room
    )
    assert req.neighbourhood_group == NeighbourhoodGroup.brooklyn
    assert req.room_type == RoomType.private_room

def test_invalid_neighbourhood_group():
    with pytest.raises(ValidationError) as exc_info:
        PredictionRequest(
            latitude=40.7,
            longitude=-73.9,
            minimum_nights=2,
            number_of_reviews=10,
            reviews_per_month=1.2,
            calculated_host_listings_count=1,
            availability_365=100,
            neighbourhood_group="InvalidPlace",
            room_type=RoomType.shared_room
        )
    assert "neighbourhood_group" in str(exc_info.value)

def test_invalid_room_type():
    with pytest.raises(ValidationError) as exc_info:
        PredictionRequest(
            latitude=40.7,
            longitude=-73.9,
            minimum_nights=2,
            number_of_reviews=10,
            reviews_per_month=1.2,
            calculated_host_listings_count=1,
            availability_365=100,
            neighbourhood_group=NeighbourhoodGroup.queens,
            room_type="Couch in living room"
        )
    assert "room_type" in str(exc_info.value)

