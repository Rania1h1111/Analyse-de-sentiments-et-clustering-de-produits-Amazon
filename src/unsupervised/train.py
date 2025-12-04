import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

from src.unsupervised.features import build_product_features
from src.unsupervised.models.base_cluster import ClusteringConfig


# =========================
# 1. Préparation des données
# =========================

def load_product_features():
    """
    Construit le DataFrame des features produits à partir
    de build_product_features().

    On suppose que le DF contient au moins :
      - mean_rating
      - n_reviews
      - satisfaction_rate
    """
    df = build_product_features()
    X = df[["mean_rating", "n_reviews", "satisfaction_rate"]].values
    return df, X


# =========================
# 2. Méthode du coude (KMeans)
# =========================

def elbow_method(X, max_k: int = 8, random_state: int = 42):
    """
    Trace la courbe inertie = f(k) pour k = 1..max_k.
    Permet de choisir visuellement un bon nombre de clusters.
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    ks = range(1, max_k + 1)

    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("Méthode du coude - Choix du nombre de clusters (KMeans)")
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Inertie intra-cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# 3. PCA + Visualisation
# =========================

def apply_pca(X_scaled, n_components: int = 2):
    """
    Applique une PCA à X_scaled et renvoie les coordonnées réduites
    ainsi que l'objet PCA.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_.sum()
    print(f"Variance expliquée par la PCA ({n_components} dim) : {var:.2f}")
    return X_reduced, pca


def plot_pca_scatter(X_reduced, labels, title: str):
    """
    Affiche un scatter plot 2D des points projetés par PCA, colorés par cluster.
    """
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="tab10")
    plt.title(title)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()


# =========================
# 4. Évaluation des clusters
# =========================

def evaluate_clustering(X_for_silhouette, labels, model_name: str) -> float:
    """
    Calcule le silhouette score pour un clustering donné.
    """
    if len(set(labels)) <= 1:
        print(f"[{model_name}] Silhouette impossible (1 seul cluster).")
        return np.nan

    score = silhouette_score(X_for_silhouette, labels)
    print(f"[{model_name}] Silhouette score : {score:.4f}")
    return score


def summarize_clusters(df, labels, model_name: str):
    """
    Affiche un résumé statistique par cluster.
    """
    df_tmp = df.copy()
    df_tmp["cluster"] = labels

    summary = (
        df_tmp.groupby("cluster")[["mean_rating", "n_reviews", "satisfaction_rate"]]
        .agg(["mean", "median", "min", "max", "count"])
    )
    print(f"\n=== Résumé des clusters ({model_name}) ===")
    print(summary)


# =========================
# 5. Pipelines KMeans et CAH
# =========================

def run_kmeans(X, df_products, config: ClusteringConfig):
    """
    Pipeline complet pour KMeans :
      - RobustScaler
      - PCA (optionnelle)
      - KMeans clustering
      - Silhouette + résumé
      - Visualisation PCA
    """
    # Mise à l'échelle robuste
    X_scaled = RobustScaler().fit_transform(X)

    # PCA éventuelle
    if config.pca_dim is not None:
        X_reduced, _ = apply_pca(X_scaled, config.pca_dim)
        X_for_clustering = X_reduced
    else:
        X_reduced = X_scaled
        X_for_clustering = X_scaled

    # Clustering
    model = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state
    )
    labels = model.fit_predict(X_for_clustering)

    # Évaluation
    evaluate_clustering(X_for_clustering, labels, "KMeans")
    summarize_clusters(df_products, labels, "KMeans")

    # Visualisation
    plot_pca_scatter(X_reduced, labels, "Projection PCA – KMeans")

    return labels, model


def run_cah(X, df_products, config: ClusteringConfig):
    """
    Pipeline complet pour CAH (AgglomerativeClustering).
    Même structure que KMeans.
    """
    # Mise à l'échelle robuste
    X_scaled = RobustScaler().fit_transform(X)

    # PCA éventuelle
    if config.pca_dim is not None:
        X_reduced, _ = apply_pca(X_scaled, config.pca_dim)
        X_for_clustering = X_reduced
    else:
        X_reduced = X_scaled
        X_for_clustering = X_scaled

    model = AgglomerativeClustering(
        n_clusters=config.n_clusters,
        linkage=config.linkage
    )
    labels = model.fit_predict(X_for_clustering)

    # Évaluation
    model_name = f"CAH ({config.linkage})"
    evaluate_clustering(X_for_clustering, labels, model_name)
    summarize_clusters(df_products, labels, model_name)

    # Visualisation
    plot_pca_scatter(X_reduced, labels, f"Projection PCA – {model_name}")

    return labels, model


# =========================
# 6. Détection des anomalies (IsolationForest)
# =========================

def detect_outliers_isolation_forest(X, df_products, config: ClusteringConfig):
    """
    Utilise IsolationForest pour repérer les produits atypiques (best-sellers, etc.).
    """
    X_scaled = RobustScaler().fit_transform(X)

    iso = IsolationForest(
        contamination=config.contamination,
        random_state=config.random_state
    )
    preds = iso.fit_predict(X_scaled)   # 1 = normal, -1 = anomalie

    df_out = df_products.copy()
    df_out["anomaly"] = preds

    anomalies = df_out[df_out["anomaly"] == -1]

    print(f"\n=== Détection d'anomalies (IsolationForest) ===")
    print(f"Nombre d'anomalies détectées : {len(anomalies)}")

    if len(anomalies) > 0:
        # On affiche les produits les plus extrêmes (ici triés par n_reviews décroissant)
        anomalies_sorted = anomalies.sort_values("n_reviews", ascending=False)
        print("\nProduits les plus atypiques :")
        cols = ["mean_rating", "n_reviews", "satisfaction_rate"]
        print(anomalies_sorted[cols].head(10))


# =========================
# 7. Comparaison globale
# =========================

def compare_methods(X, df_products):
    """
    Compare les méthodes KMeans et CAH (Ward + Average)
    avec la même configuration de base.
    """
    config = ClusteringConfig(
        n_clusters=3,
        scale=True,
        pca_dim=2,
        linkage="ward",
        random_state=42,
        contamination=0.05,
    )

    print("\n=== K-Means ===")
    run_kmeans(X, df_products, config)

    print("\n=== CAH (Ward) ===")
    config.linkage = "ward"
    run_cah(X, df_products, config)

    print("\n=== CAH (Average) ===")
    config.linkage = "average"
    run_cah(X, df_products, config)


# =========================
# 8. MAIN
# =========================

def main():
    print("=== Clustering non supervisé sur les produits ===")

    # 1. Charger les features produits
    df_products, X = load_product_features()

    # 2. Méthode du coude (facultatif, pour choisir k)
    elbow_method(X, max_k=8, random_state=42)

    # 3. Comparer KMeans / CAH
    compare_methods(X, df_products)

    # 4. Détection d'anomalies
    config = ClusteringConfig(
        n_clusters=3,
        scale=True,
        pca_dim=2,
        linkage="ward",
        random_state=42,
        contamination=0.05,
    )
    detect_outliers_isolation_forest(X, df_products, config)


if __name__ == "__main__":
    main()
