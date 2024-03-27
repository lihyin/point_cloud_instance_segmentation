# this file is extracted from the model log https://download.openmmlab.com/mmdetection3d/v1.1.0_models/cylinder3d/cylinder3d_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_144950.log

dataset_type = 'SemanticKittiDataset'
data_root = 'data/semantickitti/'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = dict({
    0: 19,
    1: 19,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: 19,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: 19,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4
})
metainfo = dict(
    classes=[
        'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
        'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
        'building', 'fence', 'vegetation', 'trunck', 'terrian', 'pole',
        'traffic-sign'
    ],
    seg_label_mapping=dict({
        0: 19,
        1: 19,
        10: 0,
        11: 1,
        13: 4,
        15: 2,
        16: 4,
        18: 3,
        20: 4,
        30: 5,
        31: 6,
        32: 7,
        40: 8,
        44: 9,
        48: 10,
        49: 11,
        50: 12,
        51: 13,
        52: 19,
        60: 8,
        70: 14,
        71: 15,
        72: 16,
        80: 17,
        81: 18,
        99: 19,
        252: 0,
        253: 6,
        254: 5,
        255: 7,
        256: 4,
        257: 4,
        258: 3,
        259: 4
    }),
    max_label=259)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'LaserMix',
            'num_areas': [3, 4, 5, 6],
            'pitch_angles': [-25, 3],
            'pre_transform': [{
                'type': 'LoadPointsFromFile',
                'coord_type': 'LIDAR',
                'load_dim': 4,
                'use_dim': 4
            }, {
                'type': 'LoadAnnotations3D',
                'with_bbox_3d': False,
                'with_label_3d': False,
                'with_seg_3d': True,
                'seg_3d_dtype': 'np.int32',
                'seg_offset': 65536,
                'dataset_type': 'semantickitti'
            }, {
                'type': 'PointSegClassMapping'
            }],
            'prob':
            1
        }],
                    [{
                        'type':
                        'PolarMix',
                        'instance_classes': [0, 1, 2, 3, 4, 5, 6, 7],
                        'swap_ratio':
                        0.5,
                        'rotate_paste_ratio':
                        1.0,
                        'pre_transform': [{
                            'type': 'LoadPointsFromFile',
                            'coord_type': 'LIDAR',
                            'load_dim': 4,
                            'use_dim': 4
                        }, {
                            'type': 'LoadAnnotations3D',
                            'with_bbox_3d': False,
                            'with_label_3d': False,
                            'with_seg_3d': True,
                            'seg_3d_dtype': 'np.int32',
                            'seg_offset': 65536,
                            'dataset_type': 'semantickitti'
                        }, {
                            'type': 'PointSegClassMapping'
                        }],
                        'prob':
                        1
                    }]],
        prob=[0.5, 0.5]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        dataset_type='semantickitti',
        backend_args=None),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        dataset_type='semantickitti',
        backend_args=None),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='SemanticKittiDataset',
        data_root='data/semantickitti/',
        ann_file='semantickitti_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype='np.int32',
                seg_offset=65536,
                dataset_type='semantickitti'),
            dict(type='PointSegClassMapping'),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'LaserMix',
                    'num_areas': [3, 4, 5, 6],
                    'pitch_angles': [-25, 3],
                    'pre_transform': [{
                        'type': 'LoadPointsFromFile',
                        'coord_type': 'LIDAR',
                        'load_dim': 4,
                        'use_dim': 4
                    }, {
                        'type': 'LoadAnnotations3D',
                        'with_bbox_3d': False,
                        'with_label_3d': False,
                        'with_seg_3d': True,
                        'seg_3d_dtype': 'np.int32',
                        'seg_offset': 65536,
                        'dataset_type': 'semantickitti'
                    }, {
                        'type': 'PointSegClassMapping'
                    }],
                    'prob':
                    1
                }],
                            [{
                                'type':
                                'PolarMix',
                                'instance_classes': [0, 1, 2, 3, 4, 5, 6, 7],
                                'swap_ratio':
                                0.5,
                                'rotate_paste_ratio':
                                1.0,
                                'pre_transform': [{
                                    'type': 'LoadPointsFromFile',
                                    'coord_type': 'LIDAR',
                                    'load_dim': 4,
                                    'use_dim': 4
                                }, {
                                    'type':
                                    'LoadAnnotations3D',
                                    'with_bbox_3d':
                                    False,
                                    'with_label_3d':
                                    False,
                                    'with_seg_3d':
                                    True,
                                    'seg_3d_dtype':
                                    'np.int32',
                                    'seg_offset':
                                    65536,
                                    'dataset_type':
                                    'semantickitti'
                                }, {
                                    'type':
                                    'PointSegClassMapping'
                                }],
                                'prob':
                                1
                            }]],
                prob=[0.5, 0.5]),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0.0, 6.28318531],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
        ],
        metainfo=dict(
            classes=[
                'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                'other-ground', 'building', 'fence', 'vegetation', 'trunck',
                'terrian', 'pole', 'traffic-sign'
            ],
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4
            }),
            max_label=259),
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=19,
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SemanticKittiDataset',
        data_root='data/semantickitti/',
        ann_file='semantickitti_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype='np.int32',
                seg_offset=65536,
                dataset_type='semantickitti',
                backend_args=None),
            dict(type='PointSegClassMapping'),
            dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
        ],
        metainfo=dict(
            classes=[
                'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                'other-ground', 'building', 'fence', 'vegetation', 'trunck',
                'terrian', 'pole', 'traffic-sign'
            ],
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4
            }),
            max_label=259),
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=19,
        test_mode=True,
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SemanticKittiDataset',
        data_root='data/semantickitti/',
        ann_file='semantickitti_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype='np.int32',
                seg_offset=65536,
                dataset_type='semantickitti',
                backend_args=None),
            dict(type='PointSegClassMapping'),
            dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
        ],
        metainfo=dict(
            classes=[
                'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                'other-ground', 'building', 'fence', 'vegetation', 'trunck',
                'terrian', 'pole', 'traffic-sign'
            ],
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4
            }),
            max_label=259),
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=19,
        test_mode=True,
        backend_args=None))
val_evaluator = dict(type='SegMetric')
test_evaluator = dict(type='SegMetric')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
grid_shape = [480, 360, 32]
model = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=[480, 360, 32],
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
            max_num_points=-1,
            max_voxels=-1)),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=[480, 360, 32],
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-05, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=128,
        num_classes=20,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0,
                         reduction='none')),
    train_cfg=None,
    test_cfg=dict(mode='whole'))
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
lr = 0.008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.008, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.1)
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
launcher = 'pytorch'
work_dir = './work_dirs/cylinder3d_8xb2-3x_lpmix_semantcikitti'
