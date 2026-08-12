_base_ = '../_base_/dior_class_suffix_base.py'

custom_imports = dict(
    imports=[
        'mmdet.evaluation.metrics.refexp_metric',
        'mmdet.models.detectors.grounding_dino_class_suffix',
        'mmdet.models.dense_heads.grounding_dino_spatial_contrast_head',
    ],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='GroundingDINOSpatialContrastHead',
        spatial_contrast_cfg=dict(
            enable=True,
            loss_weight=1.0,
            margin=0.2,
            neg_iou_thr=0.25,
            topk=10,
            score_type='logit',
            exclude_gt_iou_thr=0.7)))

# Train BERT with a 0.1 learning-rate multiplier.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'language_model': dict(lr_mult=0.1)}))

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='refexp/dior_rsvg_val_Pr@0.5'))

work_dir = 'work_dirs/dior_rsvg'
