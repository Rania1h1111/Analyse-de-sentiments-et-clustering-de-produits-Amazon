from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

MODEL_NAME = "KNN"


def build_base_model():
    return KNeighborsClassifier()


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Entraîne KNN avec GridSearchCV (5-fold CV) + évalue le meilleur modèle.

    """

    # 1) Grille d'hyperparamètres à tester
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    base_model = build_base_model()

    # 2) GridSearchCV avec validation croisée à 5 folds
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,           # 5-fold cross-validation
        n_jobs=-1,      # utilise tous les cœurs CPU
        verbose=1       # afficher la progression
    )

    print(f"\n>>> GridSearch pour {MODEL_NAME} ...")
    grid.fit(X_train, y_train)

    print(f"Meilleurs hyperparamètres pour {MODEL_NAME} : {grid.best_params_}")
    print(f"Meilleur score F1-macro (CV 5 folds) : {grid.best_score_:.4f}")

    # 3) On récupère le meilleur modèle
    best_model = grid.best_estimator_

    # 4) Prédictions sur le test
    y_pred = best_model.predict(X_test)

    print(f"\n=== {MODEL_NAME} (meilleur modèle après GridSearch) ===")
    print(classification_report(y_test, y_pred))

    # On renvoie aussi le score CV pour l'afficher dans le tableau final
    best_cv_score = grid.best_score_

    return best_model, y_pred, best_cv_score
