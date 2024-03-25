# Introduction

This repository implements the unsupervised KD-Tree DBSCAN clustering LiDAR point cloud instance segmentation algorithm from scratch. Except DBScan, we could use the graph cut and region growth algorithms as well. Here is the comparisons of the different algorithms. DBSCAN is a clustering-based algorithm. 

| Algorithm                        | Advantages                                                                 | Disadvantages                                                              |
|----------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Deep Learning-Based Segmentation | Can learn complex patterns and features in point clouds                   | Requires a large amount of labeled data for training                     |
|                                  | Highly accurate and can handle noisy data                                 | Can be computationally expensive and require powerful hardware           |
| Clustering-Based Segmentation    | Simple and easy to implement                                              | May struggle with complex geometries and irregular shapes                |
|                                  | Efficient for finding clusters in dense point clouds                      | Sensitivity to hyperparameters and starting conditions                    |
| Region Growing-Based Segmentation| Can handle noise and outliers effectively                                 | May be sensitive to initial seed points and parameters                    |
|                                  | Robust to varying densities and shapes                                    | Can be computationally expensive for large point clouds                   |
| Supervoxel-Based Segmentation    | Can effectively group points into compact and homogeneous regions         | May struggle with separating instances with overlapping supervoxels       |
|                                  | Can provide more meaningful representation of point clouds                | Sensitivity to supervoxel size and parameters                             |
| Graph-Based Segmentation         | Can capture relationships between points and model complex structures     | Computational complexity and memory requirements for constructing graphs  |
|                                  | Flexible and adaptable to different types of point cloud data             | Difficulty in defining edge weights and graph construction                |

# How to Use

1. Install with Python 3.7+:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run:

```
python implementation.py
```

Output:
<img src="./doc/dbscan.png" alt="DBSCAN Output" width="400"/>

# Limitations

Compared to sklearn's DBSCAN, the current implemented DBSCAN is slower. Ways to improve are: 1) to use depth only for DBSCAN to speed up; 2) use ball tree.

# Generate Pseudo Ground Truth by Pre-trained Cylinder3D Model

For 3D instance and panopatic segmentation, there are mainly two types of applicaiton scenarios:

* Indoor Scene Understanding: ScanNet is one of the popular benchmark datasets and PointNet++ is one of the SOTA models.
* Outdoor Driving: SemanticKITTI is one of the popular benchmark datasets and Cylinder3D is one of the SOTA outdoor point cloud panopatic segmentation model.

We use Cylinder3D to generate the pseudo ground truth:

1. [Install MMDetection3D](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)
2. Downlowd the [pre-trained Cylinder3D model and config](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/cylinder3d)
3. [Infer 3D Segmentaion by MMDetection3D](https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html#d-segmentation)

# TODO

* Evaluate the algorithm with mIoU, Panoptic Quality (PQ), and Segmentation Quality
(SQ) with the pseudo ground truth.
