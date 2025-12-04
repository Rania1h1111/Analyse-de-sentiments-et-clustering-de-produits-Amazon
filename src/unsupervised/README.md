1) Construction des features produits

À partir du dataset d’avis nettoyés, on réalise un regroupement par produit (asin).
Pour chaque produit, on extrait trois variables clés :

mean_rating → moyenne des notes attribuées

n_reviews → nombre total d’avis associés

satisfaction_rate → proportion d’avis positifs (note ≥ 4)

Chaque ligne représente donc un produit, et non un avis.
Ce format est indispensable pour analyser les produits plutôt que les utilisateurs.

Après augmentation du dataset, le nombre d'avertissements augmente (8000 avis), mais le nombre de produits reste ~53, car il s'agit du même catalogue.

2) Normalisation (RobustScaler)

Les trois variables utilisées sont sur des échelles très différentes :

note : entre 1 et 5

satisfaction : entre 0 et 1

nombre d’avis : jusqu’à 3500+ (best-seller)

 RobustScaler est utilisé plutôt que StandardScaler car :

il gère mieux les valeurs extrêmes

il utilise la médiane et l’IQR (InterQuartile Range)

il évite d’écraser les petits produits à cause d’un best-seller atypique

3) Réduction dimensionnelle (PCA)

Une PCA à 2 composantes est appliquée pour :

simplifier l’espace

réduire le bruit

faciliter les visualisations 2D

améliorer parfois la séparation des clusters

Sur ce dataset, la PCA conserve 100 % de la variance, car :

les trois variables sont fortement corrélées

PC1 ≈ importance globale du produit

PC2 ≈ caractère atypique (ex : produit très populaire ou très mal noté)

4) Méthode du coude (Elbow Method)

Avant d’appliquer K-Means, on exécute l'Elbow Method sur k = 1 → 10.

Le graphique montre un coude clair autour de :

 k = 3

Ce qui indique :

la plus grande amélioration est entre k=1 et k=3

après 3, le gain d’inertie devient marginal

il existe naturellement 3 familles de produits dans les données

→ Choisir k = 3 est donc scientifiquement justifié.

5) Clustering : K-Means & CAH

Deux méthodes complémentaires sont appliquées.

K-Means

 segmentation géométrique
 rapide, performant
 bonne structure générale

Limites :

sensible aux outliers

dépend initialisation

CAH — Classification Ascendante Hiérarchique

Deux linkages testés :

Ward : minimise la variance interne

Average : distance moyenne entre clusters

Avantages :

très bon pour détecter les produits extrêmes

peut isoler un best-seller dans un cluster à lui seul

donne des silhouettes score plus élevés dans ton cas


6) Silhouette Score

Le silhouette score est calculé pour chaque méthode :

1 → clusters très séparés

0 → clusters qui se chevauchent

négatif → mauvaise structuration



 CAH est structurellement meilleur et sépare mieux les produits.

7) Détection d’anomalies (IsolationForest)

IsolationForest est utilisé en complément des clusters pour :

détecter les produits extrêmes

repérer les best-sellers

confirmer les anomalies trouvées par CAH

Résultat :

→ 3 produits identifiés comme anomalies, dont :

un best-seller avec +3700 avis

un produit très mal noté

un produit faiblement noté + peu d'avis

Cela renforce la robustesse de l’analyse.

8) Configuration centralisée (base_cluster.py)

Le fichier base_cluster.py fournit une classe ClusteringConfig, qui centralise tous les paramètres du clustering :

n_clusters      # nombre de clusters KMeans/CAH
scale           # activer RobustScaler
pca_dim         # réduction PCA
linkage         # CAH (ward, average)
random_state    # reproductibilité
contamination   # taux d'anomalies pour IsolationForest

Pourquoi c’est utile ?

 code plus propre
 tous les modèles utilisent les mêmes réglages
 changement de paramètres en un seul endroit
 pipeline plus lisible et maintenable
 possibilité d’ajouter d’autres algorithmes très facilement

Exemple d’utilisation :
config = ClusteringConfig(n_clusters=3)

run_kmeans(X, config)
config.linkage = "ward"
run_cah(X, config)
run_isolation_forest(X, config)

9) Résultats observés (nouveau dataset augmenté)
 K-Means

3 clusters homogènes

mais moins précis sur les produits atypiques

 CAH (Ward)

meilleur silhouette score

isole parfaitement un best-seller dans un cluster de taille 1

 PCA

visualisation très claire des 3 familles de produits

 Isolation Forest

confirme les anomalies détectées par CAH

