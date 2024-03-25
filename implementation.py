# %% imports
import time
import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from algorithms.clustering_dbscan_kd_tree import DBSCANKDTree
from algorithms.sklearn_dbscan import sklearn_dbscan


# %% types
class Index(IntEnum):
    X = 0
    Y = 1
    Z = 2


# %% helper functions
def visualize_pointcloud_downsampled(
    pc: np.ndarray, downsample_factor: int = 10
) -> None:
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pc[::downsample_factor, Index.X],
        pc[::downsample_factor, Index.Y],
        pc[::downsample_factor, Index.Z],
        color="red",
        s=0.1,
    )
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.set_zlabel("z (m)", fontsize=14)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 50)
    ax.set_title("Pointcloud (3D)", fontsize=14)
    plt.show()

    # make this plot occupy 30% of the figure's width and 100% of its height
    plt.figure(figsize=(25, 25))
    plt.plot(pc[:, Index.X], pc[:, Index.Y], "rx", markersize=1, alpha=0.2)
    plt.xlabel("x (m)", fontsize=14)
    plt.ylabel("y (m)", fontsize=14)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Pointcloud (Top View)", fontsize=14)
    plt.show()


def visualize_pointcloud_downsampled_with_segment_ids(
    pc: np.ndarray, segment_ids: np.ndarray, downsample_factor: int = 10
) -> None:
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pc[::downsample_factor, Index.X],
        pc[::downsample_factor, Index.Y],
        pc[::downsample_factor, Index.Z],
        c=segment_ids[::downsample_factor],
        cmap="tab20",
        s=0.2,
    )
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.set_zlabel("z (m)", fontsize=14)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 50)
    ax.set_title("Pointcloud (3D)", fontsize=14)
    plt.show()

    # make this plot occupy 30% of the figure's width and 100% of its height
    plt.figure(figsize=(25, 25))
    plt.scatter(
        pc[:, Index.X], pc[:, Index.Y], c=segment_ids, cmap="tab20", s=1, alpha=0.5
    )
    plt.xlabel("x (m)", fontsize=14)
    plt.ylabel("y (m)", fontsize=14)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Pointcloud (Top View)", fontsize=14)
    plt.show()


# %%
pointcloud = np.load("./data/pointcloud_data.npy")
pointcloud = pointcloud[::10]

# visualize_pointcloud_downsampled(
#     pointcloud, downsample_factor=5
# )  # use 'downsample_factor=1' for no downsampling during visualization


##### TODO: REQUIRES IMPLEMENTATION ##############################
##################################################################
# input is a pointcloud of shape (N, 3)
# output is a segmentation mask of shape (N,)
# where each element is an integer representing the segment id
def segment_pointcloud(
    pointcloud: np.ndarray, algorithm="clustering_dbscan_kd_tree"
) -> np.ndarray:
    if algorithm == "sklearn_dbscan":
        labels = sklearn_dbscan(pointcloud)
    elif algorithm == "clustering_dbscan_kd_tree":
        dbscan_kd_tree = DBSCANKDTree()
        labels = dbscan_kd_tree.fit(pointcloud)
    else:
        print(
            f"Please select algorithm from 'sklearn_dbscan', 'clustering_dbscan_kd_tree'!"
        )
        exit()

    return labels


start_time = time.time()
segment_ids = segment_pointcloud(pointcloud, algorithm="clustering_dbscan_kd_tree")
end_time = time.time()
print(f"clustering_dbscan_kd_tree total time: {end_time - start_time} seconds")

start_time = time.time()
sklearn_dbscan_segment_ids = segment_pointcloud(pointcloud, algorithm="sklearn_dbscan")
end_time = time.time()
print(f"sklearn_dbscan total time: {end_time - start_time} seconds")

visualize_pointcloud_downsampled_with_segment_ids(
    pointcloud, segment_ids, downsample_factor=1
)  # use 'downsample_factor=1' for no downsampling during visualization

visualize_pointcloud_downsampled_with_segment_ids(
    pointcloud, sklearn_dbscan_segment_ids, downsample_factor=1
)  # use 'downsample_factor=1' for no downsampling during visualization
