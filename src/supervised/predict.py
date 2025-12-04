import pickle
from src.config import MODELS_DIR


def load_model_and_vectorizer():
    # adapter si le meilleur modèle change
    model_path = next(MODELS_DIR.glob("*_sentiment.pkl"))
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    print(f"Modèle chargé : {model_path.name}")
    return model, vectorizer


def predict_review(text: str):
    model, vectorizer = load_model_and_vectorizer()
    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]

    label = "SATISFAIT" if y_pred == 1 else "NON SATISFAIT"

    print("\n=== Résultat ===")
    print(f"Avis : {text}")
    print(f"Prédiction : {label}")


def main():
    print("=== Prédiction de satisfaction (supervisé) ===")
    text = input("Tape un avis en anglais :\n> ")
    predict_review(text)


if __name__ == "__main__":
    main()
