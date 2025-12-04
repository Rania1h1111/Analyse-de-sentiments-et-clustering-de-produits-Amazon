from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

MODEL_NAME = "DecisionTree"


def build_base_model():
    return DecisionTreeClassifier(random_state=42)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Entraîne un Decision Tree optimisé via GridSearchCV (5-fold CV)
    puis évalue le meilleur modèle sur le jeu de test.

    """

    # 1) Définition de la grille d'hyperparamètres
    param_grid = {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    base_model = build_base_model()

    # 2) GridSearchCV avec 5-fold cross-validation
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,               
        n_jobs=-1,
        verbose=1
    )

    print(f"\n>>> GridSearch pour {MODEL_NAME} ...")
    grid.fit(X_train, y_train)

    # Affichage des résultats CV
    print(f"Meilleurs hyperparamètres pour {MODEL_NAME} : {grid.best_params_}")
    print(f"Meilleur score F1-macro (CV 5 folds) : {grid.best_score_:.4f}")

    # 3) Meilleur modèle entraîné
    best_model = grid.best_estimator_

    # 4) Prédiction sur le jeu de test
    y_pred = best_model.predict(X_test)

    print(f"\n=== {MODEL_NAME} (meilleur modèle après GridSearch) ===")
    print(classification_report(y_test, y_pred))

    # 5) Retour du score de CV pour le tableau final
    best_cv_score = grid.best_score_

    return best_model, y_pred, best_cv_score
