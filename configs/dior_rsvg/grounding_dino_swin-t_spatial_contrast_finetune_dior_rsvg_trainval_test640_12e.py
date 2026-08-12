_base_ = './grounding_dino_swin-t_finetune_dior_rsvg_trainval_test640_val4.py'

custom_imports = dict(
    imports=[
        'mmdet.evaluation.metrics.refexp_metric',
        'mmdet.models.detectors.grounding_dino_spatial_contrast',
        'mmdet.models.dense_heads.grounding_dino_spatial_contrast_head'
    ],
    allow_failed_imports=False)

model = dict(
    type='GroundingDINOSpatialContrast',
    bbox_head=dict(
        type='GroundingDINOSpatialContrastHead',
        spatial_contrast_cfg=dict(
            enable=True,
            loss_weight=1.0,
            margin=0.2,
            neg_iou_thr=0.3,
            topk=10,
            score_type='logit',
            exclude_gt_iou_thr=0.7)))

work_dir = '/root/autodl-tmp/work_dirs/dior_rsvg_spatial_contrast_trainval_test640_12e'
