class ClusteringConfig:
    """
    Configuration générique pour les méthodes de clustering.
    Centralise tous les hyperparamètres importants.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        scale: bool = True,
        pca_dim: int | None = 2,
        linkage: str = "ward",
        random_state: int = 42,
        contamination: float = 0.05,
    ):
        # nombre de clusters pour KMeans / CAH
        self.n_clusters = n_clusters

        # appliquer une mise à l'échelle (RobustScaler)
        self.scale = scale

        # dimension de la PCA (None = pas de PCA)
        self.pca_dim = pca_dim

        # type de linkage pour CAH
        self.linkage = linkage

        # seed pour la reproductibilité
        self.random_state = random_state

        # proportion d'anomalies pour IsolationForest
        self.contamination = contamination
