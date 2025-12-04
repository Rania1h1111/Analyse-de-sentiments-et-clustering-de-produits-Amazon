import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def evaluate_single_model(y_test, y_pred, model_name: str) -> dict:
    """
    Évalue un modèle de classification binaire à partir des prédictions.
    """
    print(f"\n=== Rapport de classification : {model_name} ===")
    print(classification_report(y_test, y_pred))

    report_dict = classification_report(
        y_test, y_pred, output_dict=True
    )

    metrics = {
        "Modèle": model_name,
        "Accuracy": report_dict["accuracy"],
        "F1 macro": report_dict["macro avg"]["f1-score"],
        "Precision macro": report_dict["macro avg"]["precision"],
        "Recall macro": report_dict["macro avg"]["recall"],
    }

    return metrics


def build_summary_table(eval_rows: list[dict]) -> pd.DataFrame:
    """
    Construit un tableau récapitulatif des performances à partir
    d'une liste de dictionnaires renvoyée par evaluate_single_model
    """
    df_eval = pd.DataFrame(eval_rows)
    df_eval = df_eval.set_index("Modèle")
    return df_eval


def print_summary_table(df_eval: pd.DataFrame) -> None:
    """Affiche le tableau récapitulatif des performances."""
    print("\n=== Récapitulatif des performances (macro) ===")
    print(df_eval.round(3))


def plot_confusion_matrix(y_test, y_pred, model_name: str) -> None:
    """
    Affiche une matrice de confusion sous forme de heatmap pour un modèle donné.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non satisfait (0)", "Satisfait (1)"],
        yticklabels=["Non satisfait (0)", "Satisfait (1)"]
    )
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.tight_layout()
    plt.show()


def plot_all_confusion_matrices(predictions: dict, y_test) -> None:
    """
    Affiche une matrice de confusion pour chaque modèle
    """
    for model_name, y_pred in predictions.items():
        plot_confusion_matrix(y_test, y_pred, model_name)
