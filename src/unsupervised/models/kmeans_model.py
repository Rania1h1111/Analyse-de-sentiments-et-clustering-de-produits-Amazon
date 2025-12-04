# src/unsupervised/models/kmeans_model.py

from sklearn.cluster import KMeans

MODEL_NAME = "KMeans"

def build_model(n_clusters=3, random_state=42):
    return KMeans(n_clusters=n_clusters, random_state=random_state)

def train_and_predict(model, X_scaled):
    labels = model.fit_predict(X_scaled)
    return model, labels
