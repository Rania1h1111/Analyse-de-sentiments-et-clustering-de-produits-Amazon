import re
from pathlib import Path

import pandas as pd


# Chemin vers le fichier brut Kaggle
RAW_FILE = Path("data/raw/amazon_reviews.csv")

# Chemin de sortie des données préparées
OUTPUT_FILE = Path("data/processed/amazon_reviews_clean.csv")

# NOMS DES COLONNES 
TEXT_COL = "reviews.text"       # texte de l'avis
RATING_COL = "reviews.rating"   # note 
PRODUCT_ID_COL = "asins"        # identifiant produit


def clean_text(text: str) -> str:
    """
    Nettoie un texte d'avis :
    - passe en minuscules
    - enlève les URLs
    - enlève la ponctuation et les chiffres
    - supprime les espaces multiples
    """
    if pd.isna(text):
        return ""

    # minuscules
    text = text.lower()

    # suppression des urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # garder seulement lettres et espaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # espaces multiples -> simple
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_prepare_data(n_rows: int | None = None) -> pd.DataFrame:

    print(f"Chargement des données depuis {RAW_FILE} ...")
    df = pd.read_csv(RAW_FILE, nrows=n_rows)
    print(f"Shape brut : {df.shape}")

    print("\nColonnes trouvées :")
    print(df.columns.tolist())

    # Vérifier que les colonnes nécessaires existent
    for col in [TEXT_COL, RATING_COL]:
        if col not in df.columns:
            raise ValueError(
                f"La colonne '{col}' n'existe pas dans le CSV. "
                f"Colonnes disponibles : {df.columns.tolist()}"
            )

    # Colonnes qu'on garde
    cols_to_keep = [TEXT_COL, RATING_COL]
    if PRODUCT_ID_COL in df.columns:
        cols_to_keep.append(PRODUCT_ID_COL)


    df = df[cols_to_keep].copy()

    # Supprimer les lignes sans texte ou sans note
    df = df.dropna(subset=[TEXT_COL, RATING_COL])

    # Convertir la note en numérique
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=[RATING_COL])

    # Filtrer les notes hors [1, 5] (au cas où)
    df = df[(df[RATING_COL] >= 1) & (df[RATING_COL] <= 5)]

    # Création du label binaire : 1 = satisfait, 0 = pas satisfait
    # ici : 4 et 5 -> satisfait, 0 à 3 -> pas satisfait
    df["label"] = (df[RATING_COL] >= 4).astype(int)

    print(df["label"].value_counts(normalize=True))

    # Nettoyage du texte
    df["text_clean"] = df[TEXT_COL].astype(str).apply(clean_text)

    # Enlever les lignes dont le texte nettoyé est vide
    df = df[df["text_clean"].str.len() > 0]

    return df


def save_prepared_data(df: pd.DataFrame, output_path: Path = OUTPUT_FILE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDonnées préparées enregistrées dans : {output_path}")


def main():
    df_clean = load_and_prepare_data(n_rows=None)

    print("\nAperçu des données préparées :")
    print(df_clean.head())

    save_prepared_data(df_clean)


if __name__ == "__main__":
    main()
