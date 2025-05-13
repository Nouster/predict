# Introduction

Après avoir récupéré le dataset sur Kaggle, puis opérer certaines transformations sur Google Collab, j'ai exporté le modèle dans le projet. J’ai tenté diverses approches pour améliorer mon R2 et limiter les erreurs quadratiques. J’ai ensuite utilisé un modèle GradientBoostingRegressor pour entrainer mon modèle que j’ai ensuite intégré au projet.

## Airbnb Price Predictor – CLI Tool

Le fichier predict.py me permet de prédire le prix d’une location Airbnb à New York en se basant sur plusieurs caractéristiques renseignées par l'utilisateur via la console.

## Tests unitaires

Le premier test (test_valid_input()) a plusieurs objectifs : - vérifier que le modèle accepte les bonnes données. - vérifier que le type de retour est conforme (ici, un tableau numpy).

Ce test me sert de référence. S’il échoue, ça signifie soit que les
colonnes sont mal préparées, soit que le modèle a un problème fondamental.

Le second test (test_empty_input()) me permet de m'assurer que le modèle réagit correctement face à des données manquantes.
Les objectis de ce test : - Je vérifie que le modèle réagit correctement face à des données manquantes ou vides. - Je m'attend à une ValueError (le modèle ne peut pas convertir les chaînes vides en float)

_Ce test est important car il me garantit que le modèle ne retourne pas des résultats silencieusement incorrect._

## Test pour l'API

Les deux derniers tests sont là pour s'assurer que les champs neighbourhood_group et room_type dans la classe PredictionRequest acceptent uniquement les valeurs que j'ai définies dans l’enum, et lèvent une exception sinon.

## Pour lancer l'API

uvicorn main:app --reload
