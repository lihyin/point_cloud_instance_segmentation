from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def sklearn_dbscan(point_cloud):
    # Normalize the data
    scaler = StandardScaler()
    point_cloud_normalized = scaler.fit_transform(point_cloud)

    # Perform DBSCAN clustering based on depth
    dbscan = DBSCAN(eps=0.1, min_samples=10, algorithm="auto")
    clusters = dbscan.fit_predict(point_cloud_normalized)

    return clusters
