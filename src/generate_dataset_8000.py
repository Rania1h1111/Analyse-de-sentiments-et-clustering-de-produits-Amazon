import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_AMAZON_FILE_1000, PROCESSED_DATA_DIR


TARGET_SIZE = 8000
OUTPUT_FILE = PROCESSED_DATA_DIR / "amazon_reviews_8000.csv"


EXTRA_TOKENS = [
    "really", "very", "super",
    "honestly", "actually", "amazingly",
    "highly", "strongly"
]


def augment_text(text: str) -> str:
    """
    Petite augmentation de texte très simple :
    - on insère un mot comme 'really', 'very', etc.
    - parfois on enlève un mot au hasard
    Objectif : créer une variante légère sans changer le sens global.
    """
    words = text.split()
    if len(words) < 4:
        return text

    # Copie locale
    new_words = words.copy()

    # 1) Ajouter un adverbe à un endroit aléatoire
    insert_idx = random.randint(1, len(new_words) - 1)
    token = random.choice(EXTRA_TOKENS)
    new_words.insert(insert_idx, token)

    # 2) retirer un mot aléatoire
    if len(new_words) > 6 and random.random() < 0.5:
        remove_idx = random.randint(0, len(new_words) - 1)
        new_words.pop(remove_idx)

    return " ".join(new_words)


def main():
    print(f"Chargement du dataset nettoyé depuis : {PROCESSED_AMAZON_FILE_1000}")
    df = pd.read_csv(PROCESSED_AMAZON_FILE_1000)
    n_current = len(df)
    print(f"Nombre de lignes actuel : {n_current}")

    # === CAS 1 : le dataset est déjà plus grand que 8000 ===
    if n_current >= TARGET_SIZE:
        print(f"Le dataset contient déjà {n_current} lignes.")
        print(f"Échantillonnage aléatoire de {TARGET_SIZE} lignes...")
        df_8000 = df.sample(TARGET_SIZE, random_state=42).reset_index(drop=True)
        df_8000.to_csv(OUTPUT_FILE, index=False)
        print(f"Dataset final enregistré dans : {OUTPUT_FILE}")
        print(df_8000.shape)
        return

    # === CAS 2 : le dataset est plus petit que 8000 → on augmente ===
    print("Dataset plus petit que 8000 lignes, on génère des lignes supplémentaires...")
    remaining = TARGET_SIZE - n_current
    print(f"Nombre de lignes à générer : {remaining}")

    # On va échantillonner des lignes existantes avec remise
    indices = np.random.choice(n_current, size=remaining, replace=True)
    rows = df.iloc[indices].copy()

    # On applique une augmentation sur la colonne text_clean
    if "text_clean" not in rows.columns:
        raise ValueError(
            "La colonne 'text_clean' est absente du dataset. "
            "Assure-toi d'avoir lancé data_preparation avant."
        )

    print("Application de l'augmentation de texte sur les lignes dupliquées...")
    rows["text_clean"] = rows["text_clean"].astype(str).apply(augment_text)


    # Concat original + augmentée
    df_aug = pd.concat([df, rows], ignore_index=True)

    # Juste pour être sûr : truncation/échantillonnage à 8000 si jamais on dépasse
    if len(df_aug) > TARGET_SIZE:
        df_aug = df_aug.sample(TARGET_SIZE, random_state=42).reset_index(drop=True)

    print(f"Taille finale du dataset : {len(df_aug)} lignes")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset de 8000 lignes enregistré dans : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
