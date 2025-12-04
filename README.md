1. Contexte

Ce projet a été réalisé dans le cadre du module Techniques d’Apprentissage Artificiel (TAA) du Master 1 Informatique.

Il met en place un pipeline complet de Machine Learning, comprenant :

préparation et nettoyage des données

création d’un dataset étendu (8000 lignes)

apprentissage supervisé (classification du sentiment)

apprentissage non supervisé (clustering de produits)

visualisation et interprétation des résultats

sauvegarde et utilisation réelle du modèle

Le projet combine deux angles complémentaires :

 Analyse de sentiment → prédire automatiquement si un avis est positif ou négatif.
 Segmentation produit → regrouper les produits Amazon selon leurs performances globales.

2. Problématique

L’objectif est d’analyser automatiquement des avis Amazon afin de :

1) Classification supervisée (NLP)

prédire si un avis exprime un sentiment positif (1) ou négatif (0)

utiliser TF-IDF + modèles ML optimisés

comparer plusieurs algorithmes

2) Clustering non supervisé

regrouper les produits selon :

leur note moyenne

leur nombre d’avis

leur taux de satisfaction

identifier les produits :

 best-sellers

 atypiques (anomalies)

 moyens



3. Préparation des données
 Étape 1 — Nettoyage du dataset original
python -m src.data_preparation


Ce script :

nettoie le texte (text_clean)

supprime les valeurs manquantes

convertit les notes en numérique

crée une étiquette binaire label :

1 si note ≥ 4

0 sinon

génère amazon_reviews_clean.csv (≈ 1000 avis)

 Étape 2 — Génération d’un dataset étendu (8000 lignes)
python -m src.generate_dataset_8000


Ce script :

charge les 1000 avis d’origine

génère automatiquement 7000 avis augmentés :

synonymes

reformulations

permutations de phrases

remplacement de mots neutres

mélange et sauvegarde dans :

data/processed/amazon_reviews_8000.csv


 Rôle dans le projet :

entraîner les modèles supervisés sur plus de données

obtenir des métriques plus stables et plus proches d’une situation réelle

conserver toujours les 1000 avis d’origine dans le test, pour une évaluation honnête

4. Apprentissage supervisé — Classification

Exécution :

python -m src.supervised.train


Le pipeline réalise :

vectorisation TF-IDF (5000 features, uni/bi-grammes)

split :

1000 avis originaux → toujours dans le test

reste des 8000 → train & test complémentaires

optimisation des modèles via GridSearchCV (5-fold)

comparaison :

Accuracy

Precision macro

Recall macro

F1-macro (métrique principale)

matrices de confusion

barplot comparatif des scores

sauvegarde du meilleur modèle :

models/RandomForest_sentiment.pkl
models/tfidf_vectorizer.pkl

Tester une prédiction
python -m src.supervised.predict

5. Apprentissage non supervisé — Clustering

Exécution :

python -m src.unsupervised.train


Pipeline :

 Étape 1 — Construction des features produits

Pour chaque produit :

note moyenne

nombre d’avis

taux de satisfaction

(53 produits après regroupement)

 Étape 2 — Normalisation

StandardScaler

 Étape 3 — PCA

Réduction à 2 composantes pour visualiser la structure.

 Étape 4 — Clustering

K-Means (k = 3)

CAH (Ward + Average)

 Étape 5 — Visualisations

Scatter plots PCA

Résumés statistiques des clusters

Silhouette Score

Détection d’anomalies (IsolationForest)

Résultats typiques :

Cluster principal : produits moyens

Cluster secondaire : produits très bien notés

Cluster isolé : best-seller extrême

ex : +3500 avis, satisfaction 92 %

La CAH (Ward) identifie systématiquement ce produit atypique dans un cluster de taille 1.

6. Exécution complète du pipeline
Installer les dépendances
pip install -r requirements.txt

Nettoyage du dataset originel
python -m src.data_preparation

Génération d’un dataset étendu (8000 lignes)
python -m src.generate_dataset_8000

Classification supervisée
python -m src.supervised.train
python -m src.supervised.predict

Clustering non supervisé
python -m src.unsupervised.train