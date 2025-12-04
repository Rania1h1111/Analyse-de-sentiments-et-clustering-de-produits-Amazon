import pandas as pd
from src.config import PROCESSED_AMAZON_FILE_8000

def build_product_features():
    df = pd.read_csv(PROCESSED_AMAZON_FILE_8000)

    grouped = df.groupby("asins").agg(
        mean_rating=("reviews.rating", "mean"),
        n_reviews=("reviews.rating", "count"),
        satisfaction_rate=("label", "mean")
    ).reset_index()

    return grouped
