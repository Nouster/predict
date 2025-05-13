import pandas as pd
import pickle

# Charger le modèle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialiser le dictionnaire avec toutes les colonnes à zéro
input_data = {
    'latitude': 0.0,
    'longitude': 0.0,
    'minimum_nights': 0,
    'number_of_reviews': 0,
    'reviews_per_month': 0.0,
    'calculated_host_listings_count': 0,
    'availability_365': 0,
    'neighbourhood_group_Brooklyn': 0,
    'neighbourhood_group_Manhattan': 0,
    'neighbourhood_group_Queens': 0,
    'neighbourhood_group_Staten Island': 0,
    'room_type_Private room': 0,
    'room_type_Shared room': 0
}

# Saisie utilisateur pour les variables numériques
input_data['latitude'] = float(input("Latitude : "))
input_data['longitude'] = float(input("Longitude : "))
input_data['minimum_nights'] = int(input("Minimum nights : "))
input_data['number_of_reviews'] = int(input("Number of reviews : "))
input_data['reviews_per_month'] = float(input("Reviews per month : "))
input_data['calculated_host_listings_count'] = int(input("Host listings count : "))
input_data['availability_365'] = int(input("Availability (jours/an) : "))

# Saisie pour neighbourhood_group (One-hot)
neighbourhood = input("Neighbourhood group (Brooklyn, Manhattan, Queens, Staten Island) : ").strip()
neighbourhood_key = f'neighbourhood_group_{neighbourhood}'
if neighbourhood_key in input_data:
    input_data[neighbourhood_key] = 1
else:
    print("❌ Valeur de neighbourhood non reconnue, aucune case cochée.")

# Saisie pour room_type (One-hot)
room_type = input("Room type (Private room, Shared room, Entire home/apt) : ").strip()
if room_type == "Private room":
    input_data['room_type_Private room'] = 1
elif room_type == "Shared room":
    input_data['room_type_Shared room'] = 1
# Pas besoin d’ajouter 'Entire home/apt' car c’est la catégorie de base implicite (quand les autres = 0)

# Convertir en DataFrame avec les colonnes dans le bon ordre
input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

# Prédiction
prediction = model.predict(input_df)
print("💰 Prix prédit :", prediction[0])
