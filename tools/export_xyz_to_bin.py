import numpy as np

pointcloud = np.load("./data/pointcloud_data.npy")
pointcloud = pointcloud[::1]  # downsample by 1

points = np.zeros([len(pointcloud), 4], dtype=np.float32)
points[:, 0] = pointcloud[:, 0]
points[:, 1] = pointcloud[:, 1]
points[:, 2] = pointcloud[:, 2]
# points[:, 3] is intensity and leave it as 0

with open("./data/pointcloud_data_4d_downsample1.bin", "wb") as f:
    f.write(points.tobytes())
