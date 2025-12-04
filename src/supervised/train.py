import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_AMAZON_FILE_1000,
    PROCESSED_AMAZON_FILE_8000,
    MODELS_DIR,
)
from src.supervised.models import ALL_MODELS
from src.supervised.evaluation import (
    evaluate_single_model,
    build_summary_table,
    print_summary_table,
    plot_all_confusion_matrices,
)


# =======================
# 1. Chargement des données
# =======================

def load_original_test():
    """Charge les 1000 avis d'origine (toujours dans le test)."""
    df_orig = pd.read_csv(PROCESSED_AMAZON_FILE_1000)
    print(f"[TEST ORIG] {PROCESSED_AMAZON_FILE_1000} -> {df_orig.shape}")
    return df_orig


def load_augmented():
    """Charge le dataset augmenté (8000)."""
    df_aug = pd.read_csv(PROCESSED_AMAZON_FILE_8000)
    print(f"[AUG] {PROCESSED_AMAZON_FILE_8000} -> {df_aug.shape}")
    return df_aug


# =======================
# 2. Construction Train / Test
# =======================

def build_train_test():
    """
    Objectif :
      - Les 1000 lignes d'origine sont TOUJOURS dans le test.
      - On peut ajouter d'autres lignes (augmentées) dans le test.
      - Le train = reste du dataset augmenté.

    On identifie les originaux dans le dataset augmenté grâce à 'text_clean'.
    """

    df_orig = load_original_test()
    df_aug = load_augmented()

    # On s'assure d'avoir 'text_clean' et 'label'
    for col in ["text_clean", "label"]:
        if col not in df_orig.columns or col not in df_aug.columns:
            raise ValueError(f"Colonne manquante dans les datasets : {col}")

    # --- On commence par mettre les 1000 originaux dans le test
    df_test_orig = df_orig.copy()

    # --- On enlève du dataset augmenté toutes les lignes dont le texte
    #     est exactement dans le dataset original, pour éviter de les avoir en train.
    orig_texts = set(df_test_orig["text_clean"].astype(str).unique())
    mask_is_original = df_aug["text_clean"].astype(str).isin(orig_texts)
    df_aug_rest = df_aug[~mask_is_original].copy()

    print(f"[INFO] Lignes dans df_aug marquées comme 'orig' : {mask_is_original.sum()}")
    print(f"[INFO] Taille du reste (augmenté sans originaux) : {df_aug_rest.shape}")

    # --- on ajoute un peu de df_aug_rest dans le test (en plus des 1000)
    # par exemple 20 % du reste dans le test.
    if len(df_aug_rest) > 0:
        df_train_part, df_test_extra = train_test_split(
            df_aug_rest,
            test_size=0.2,
            random_state=42,
            stratify=df_aug_rest["label"],
        )
    else:
        df_train_part = df_aug_rest
        df_test_extra = pd.DataFrame(columns=df_aug_rest.columns)

    # --- Test final = originaux + extra dupliqués
    df_test = pd.concat([df_test_orig, df_test_extra], ignore_index=True)

    # --- Train final = le reste
    df_train = df_train_part.copy()

    print(f"[FINAL] Train shape : {df_train.shape}")
    print(f"[FINAL] Test shape  : {df_test.shape} (dont {df_test_orig.shape[0]} originaux)")

    X_train_text = df_train["text_clean"]
    y_train = df_train["label"]
    X_test_text = df_test["text_clean"]
    y_test = df_test["label"]

    return X_train_text, X_test_text, y_train, y_test


# =======================
# 3. Vectorisation TF-IDF
# =======================

def vectorize_text(X_train_text, X_test_text):
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    return X_train, X_test, tfidf


# =======================
# 4. Entraînement des modèles
# =======================

def train_all_models(X_train, y_train, X_test, y_test):
    results = {}
    eval_rows = []
    predictions = {}

    for model_module in ALL_MODELS:
        name = model_module.MODEL_NAME
        print(f"\n>>> Entraînement du modèle : {name}")

        model, y_pred, best_cv_score = model_module.train_and_evaluate(
            X_train, y_train, X_test, y_test
        )

        results[name] = model
        predictions[name] = y_pred

        metrics = evaluate_single_model(y_test, y_pred, name)
        metrics["F1 CV (GridSearch)"] = best_cv_score
        eval_rows.append(metrics)

    return results, eval_rows, predictions


# =======================
# 5. Sauvegarde du meilleur modèle
# =======================

def save_best_model(best_name, best_model, tfidf):
    model_path = MODELS_DIR / f"{best_name}_sentiment.pkl"
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(tfidf, f)

    print(f"Modèle sauvegardé dans : {model_path}")
    print(f"Vectoriseur TF-IDF sauvegardé dans : {vec_path}")


# =======================
# 6. MAIN
# =======================

def main():
    # --- Construction train/test 
    X_train_text, X_test_text, y_train, y_test = build_train_test()

    # --- TF-IDF
    X_train, X_test, tfidf = vectorize_text(X_train_text, X_test_text)

    # --- Entraînement + évaluation
    results, eval_rows, predictions = train_all_models(
        X_train, y_train, X_test, y_test
    )

    # --- Tableau comparatif
    df_eval = build_summary_table(eval_rows)
    print_summary_table(df_eval)

    # --- Matrices de confusion
    plot_all_confusion_matrices(predictions, y_test)

    # --- Barplot comparatif
    models = df_eval.index.tolist()
    accuracy = df_eval["Accuracy"].tolist()
    precision = df_eval["Precision macro"].tolist()
    recall = df_eval["Recall macro"].tolist()
    f1_score = df_eval["F1 macro"].tolist()

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
    plt.bar(x - 0.5*width, precision, width, label='Precision')
    plt.bar(x + 0.5*width, recall, width, label='Recall')
    plt.bar(x + 1.5*width, f1_score, width, label='F1-Score')

    plt.ylabel('Score')
    plt.title('Comparaison des modèles - Métriques de Classification')
    plt.xticks(x, models)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Sélection du meilleur modèle (F1-macro)
    best_name = df_eval["F1 macro"].idxmax()
    best_model = results[best_name]
    best_f1 = df_eval.loc[best_name, "F1 macro"]

    print(f"\n>> Meilleur modèle : {best_name} (F1-macro = {best_f1:.4f})")

    save_best_model(best_name, best_model, tfidf)


if __name__ == "__main__":
    main()
