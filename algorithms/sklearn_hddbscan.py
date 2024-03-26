from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler


def sklearn_hdbscan(point_cloud):
    # Normalize the data
    scaler = StandardScaler()
    point_cloud_normalized = scaler.fit_transform(point_cloud)

    # Perform HDDBSCAN clustering based on depth
    hdbscan = HDBSCAN()
    clusters = hdbscan.fit_predict(point_cloud_normalized)

    return clusters
