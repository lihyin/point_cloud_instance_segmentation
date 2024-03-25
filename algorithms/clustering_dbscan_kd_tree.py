import numpy as np  # import the numpy library for array handling
from sklearn.preprocessing import (
    StandardScaler,
)  # import standard scaler for feature scaling


# Define the main class for DBSCAN using a KD-Tree approach
class DBSCANKDTree:
    # Class constructor with default epsilon and minimum points parameters
    def __init__(self, eps=0.1, min_points=3):
        self.eps = eps  # Epsilon value for the neighborhood search
        self.min_points = (
            min_points  # Minimum number of points required to form a cluster
        )

    # Fit function for the model, where 'X' is the dataset
    def fit(self, X):
        # Initialize the standard scaler and scale the dataset
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Store the scaled dataset and dataset size
        self.X = X
        self.n_points, _ = X.shape

        # Initialize arrays to track visited points and cluster IDs
        self.visited = np.zeros(self.n_points)
        self.clusters = np.zeros(self.n_points)
        self.current_cluster = 1  # Start with cluster ID 1

        # Initialize the KD-Tree with the dataset
        kd_tree = KDTree(X)

        # Iterate through all points to identify clusters
        for i in range(self.n_points):
            # If this point has not been visited
            if self.visited[i] == 0:
                self.visited[i] = 1  # Mark this point as visited
                # Find neighbors within epsilon radius
                neighbors = kd_tree.get_neighbors_within_radius(
                    X[i].reshape(1, -1), r=self.eps
                )

                # If the number of neighbors is below the threshold, mark as noise
                if len(neighbors) < self.min_points:
                    self.clusters[i] = -1
                else:
                    # Otherwise, expand the cluster from this point
                    self.expand_cluster(kd_tree, i, neighbors)
                    self.current_cluster += (
                        1  # Increment cluster ID for the next cluster
                    )

        # Return the array of cluster IDs for each point
        return self.clusters

    # Function to expand clusters from a given point
    def expand_cluster(self, kd_tree, point_idx, neighbors):
        # Assign the current cluster ID to the initial point
        self.clusters[point_idx] = self.current_cluster

        # Iterate through each neighbor
        for neighbor in neighbors:
            # If this neighbor hasn't been visited
            if self.visited[neighbor] == 0:
                self.visited[neighbor] = 1  # Mark it as visited
                # Find neighbors of this neighbor point
                new_neighbors = kd_tree.get_neighbors_within_radius(
                    self.X[neighbor].reshape(1, -1), r=self.eps
                )

                # If this neighbor is also a core point, add its neighbors to the list
                if len(new_neighbors) >= self.min_points:
                    neighbors = np.concatenate((neighbors, new_neighbors))

            # If the neighbor doesn't belong to any existing cluster
            if self.clusters[neighbor] == 0:
                self.clusters[neighbor] = (
                    self.current_cluster
                )  # Assign it to the current cluster


# KD-Tree class definition
class KDTree:
    # Constructor of the KD-Tree with the dataset points
    def __init__(self, points):
        # Add the original index for each point
        new_points = []
        for i in range(len(points)):
            new_points.append(np.append(points[i], [i]))

        self.points = new_points
        self.tree = self.build_kdtree(new_points)  # Build the KD-Tree

    # Recursive function to build a KD-Tree
    def build_kdtree(self, points, depth=0):
        if len(points) == 0:
            return None

        k = (
            len(points[0]) - 1
        )  # Remove index from consideration to get the number of dimensions
        axis = (
            depth % k
        )  # Choose axis based on depth so we cycle through all dimensions

        # Sort points and find median to determine where to split the dataset
        points = sorted(points, key=lambda x: x[axis])
        median = len(points) // 2

        # Build a dictionary representing the node and recursive subtrees
        return {
            "point": points[median],  # Point at the median
            "left": self.build_kdtree(points[:median], depth + 1),  # Left subtree
            "right": self.build_kdtree(
                points[median + 1 :], depth + 1
            ),  # Right subtree
        }

    # Get all neighbors within a given radius from the target point
    def get_neighbors_within_radius(self, target, r):
        self.radius_neighbors = []  # Reset the list of neighbors
        # Helper function to recursively search for neighbors within the radius
        self.get_neighbors_within_radius_helper(self.tree, target, r, 0)
        return self.radius_neighbors

    # Recursive helper function to find neighbors within the specified radius
    def get_neighbors_within_radius_helper(self, node, target, r, depth):
        if node is not None:  # If the node exists
            # Get the dimension to compare based on depth
            k = len(target[0])
            axis = depth % k

            # Check if point in node could be within the target radius in the given axis
            if (
                node["point"][axis] >= target[0][axis] - r
                and node["point"][axis] <= target[0][axis] + r
            ):
                # Check distance and if point is within radius, add to list of neighbors
                if np.linalg.norm(node["point"][:3] - target) <= r:
                    self.radius_neighbors.append(
                        int(node["point"][-1])  # the last element is the original point index
                    )

                # Check left and right child nodes if they could contain points within radius
                self.get_neighbors_within_radius_helper(
                    node["left"], target, r, depth + 1
                )
                self.get_neighbors_within_radius_helper(
                    node["right"], target, r, depth + 1
                )
            # Decide whether to traverse left or right subtree based on axis boundary
            elif node["point"][axis] > target[0][axis] + r:
                self.get_neighbors_within_radius_helper(
                    node["left"], target, r, depth + 1
                )
            else:
                self.get_neighbors_within_radius_helper(
                    node["right"], target, r, depth + 1
                )
