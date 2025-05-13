import pytest
import pandas as pd
import pickle
import numpy as np

# Charger un modèle fictif pour les tests
@pytest.fixture(scope='module')
def model():
    # Tu dois charger ton modèle réel ici.
    return pickle.load(open('model.pkl', 'rb'))  # À adapter

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
    assert isinstance(prediction, np.ndarray)  # Vérifie que la prédiction est un tableau numpy

def test_empty_input(model, feature_names):
    """Test avec une entrée vide"""
    input_data = {feature: '' for feature in feature_names}
    input_df = pd.DataFrame([input_data], columns=feature_names)
    with pytest.raises(ValueError):  # Vérifie si une erreur est levée
        model.predict(input_df)

def test_special_characters_input(model, feature_names):
    """Test avec des caractères spéciaux dans les inputs"""
    input_data = {
        'latitude': '40.7128@',  # Caractère spécial
        'longitude': -74.0060,
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

