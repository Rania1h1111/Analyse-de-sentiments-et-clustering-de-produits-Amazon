import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.decomposition import PCA

def plot_pca(X_scaled, labels, model_name):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_pca[:,0], y=X_pca[:,1],
        hue=labels,
        palette="Set1",
        s=70
    )
    plt.title(f"Projection PCA – {model_name}")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend(title="Cluster")
    plt.show()


def plot_cluster_summary(df_with_clusters, model_name):
    print(f"\n=== Résumé des clusters ({model_name}) ===")
    print(
        df_with_clusters.groupby("cluster")[
            ["mean_rating", "n_reviews", "satisfaction_rate"]
        ].mean()
    )
