from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

MODEL_NAME = "RandomForest"


def build_base_model():
    return RandomForestClassifier(random_state=42)


def train_and_evaluate(X_train, y_train, X_test, y_test):

    # 1) Grille d'hyperparamètres
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    base_model = build_base_model()

    # 2) GridSearchCV avec validation croisée à 5 folds
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,               # validation croisée en 5 folds
        n_jobs=-1,
        verbose=1
    )

    print(f"\n>>> GridSearch pour {MODEL_NAME} ...")
    grid.fit(X_train, y_train)

    # 3) Affichage des résultats CV
    print(f"Meilleurs hyperparamètres pour {MODEL_NAME} : {grid.best_params_}")
    print(f"Meilleur score F1-macro (CV 5 folds) : {grid.best_score_:.4f}")

    # 4) Meilleur modèle trouvé
    best_model = grid.best_estimator_

    # 5) Prédictions sur le test
    y_pred = best_model.predict(X_test)

    print(f"\n=== {MODEL_NAME} (meilleur modèle après GridSearch) ===")
    print(classification_report(y_test, y_pred))

    # 6) Score CV renvoyé pour tableau final
    best_cv_score = grid.best_score_

    return best_model, y_pred, best_cv_score
