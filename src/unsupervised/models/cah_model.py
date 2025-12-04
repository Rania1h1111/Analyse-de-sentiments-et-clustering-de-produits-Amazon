# src/unsupervised/models/cah_model.py

from sklearn.cluster import AgglomerativeClustering

MODEL_NAME = "CAH"

def build_model(n_clusters=3, linkage="ward"):
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )

def train_and_predict(model, X_scaled):
    labels = model.fit_predict(X_scaled)
    return model, labels
