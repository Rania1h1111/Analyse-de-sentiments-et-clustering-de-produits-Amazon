1. Objectif

L’objectif du module supervisé est de prédire automatiquement la satisfaction d’un avis Amazon à partir de son contenu textuel.

1 → avis satisfait

0 → avis insatisfait

Le meilleur modèle est sélectionné automatiquement selon la métrique F1-macro, puis sauvegardé afin d’être utilisé pour analyser de nouveaux avis via le script predict.py.

2. Structure du module supervisé
src/supervised/
│
├── train.py                 # pipeline complet : train/test, GridSearch, évaluation, sauvegarde
├── predict.py               # prédiction en direct d’un avis saisi au clavier
├── evaluation.py            # métriques, tableaux comparatifs, matrices de confusion
└── models/
    ├── knn_model.py          # KNN + hyperparamètres optimisés via GridSearchCV
    ├── decision_tree_model.py   # Arbre de décision + GridSearchCV
    └── random_forest_model.py   # Random Forest + GridSearchCV

 Remarques

Chaque fichier dans models/ contient une fonction train_and_evaluate(...).

train.py orchestre tout le pipeline.

predict.py charge le meilleur modèle pour une utilisation réelle.

3. Pipeline supervisé — Détaillé et clair
1) Chargement des données

Les données nettoyées sont chargées depuis :

data/processed/amazon_reviews_clean.csv


Ce fichier est produit par :

python -m src.data_preparation


Colonnes utilisées :

text_clean → texte nettoyé

label → 1 si note ≥ 4, sinon 0

2) Vectorisation TF-IDF

Transforme le texte en vecteurs numériques.

Paramètres utilisés :

max_features = 5000

ngram_range = (1, 2) → uni-grammes + bi-grammes

stop_words = "english"

Le TF-IDF est fité sur le train puis sauvegardé pour predict.py.

3) Split train/test

Séparation du dataset :

80% pour l’apprentissage

20% pour le test

Avec :

train_test_split(..., test_size=0.2, stratify=y, random_state=42)


 stratify=y conserve la proportion de classes
 reproductibilité (random_state=42)

4) Entraînement des modèles

Pour chaque modèle dans src/supervised/models/, on lance :

model, y_pred, best_cv_score = model_module.train_and_evaluate(
    X_train, y_train, X_test, y_test
)


Chaque modèle :

construit une version de base

applique GridSearchCV (5-fold)

trouve les meilleurs hyperparamètres

réentraîne le meilleur modèle

renvoie :

best_model

y_pred sur le test

best_cv_score (F1-macro CV)

Les prédictions servent ensuite pour :

les métriques

la comparaison finale

les matrices de confusion

5) Évaluation

L’évaluation est gérée dans evaluation.py.

Pour chaque modèle, on calcule :

Accuracy

Precision macro

Recall macro

F1 macro

Deux types de sorties :

 Rapport individuel

Affiché dans la console (classification_report).

 Tableau récapitulatif

Consolidé avec :

build_summary_table(eval_rows)

 Visualisations

matrices de confusion

barplot comparatif (Accuracy, Precision, Recall, F1)

(optionnel) radar chart

6) Sélection et sauvegarde du meilleur modèle

Le modèle ayant la meilleure F1-macro est automatiquement sélectionné.

Il est enregistré sous :

models/<NomModele>_sentiment.pkl


Ainsi que le TF-IDF sous :

models/tfidf_vectorizer.pkl


Ces fichiers sont utilisés dans predict.py.

4. Rôle du script predict.py

predict.py simule une utilisation réelle du modèle.

Il permet :

de charger le meilleur modèle sauvegardé

de charger le TF-IDF

de saisir un avis dans le terminal

d’obtenir une prédiction instantanée :

 “SATISFAIT”
ou
 “NON SATISFAIT”

C’est la partie « application » du pipeline.

5. Utilisation de GridSearchCV

Pour chaque modèle, GridSearchCV (CV=5) est utilisé afin de :

tester plusieurs hyperparamètres

maximiser F1-macro

éviter le sur-apprentissage

obtenir un modèle beaucoup plus performant qu’un modèle par défaut

C’est un élément essentiel d’un pipeline professionnel.

6. Commandes essentielles
 Entraînement complet + comparaison + sauvegarde
python -m src.supervised.train

 Prédiction sur un avis utilisateur
python -m src.supervised.predict
