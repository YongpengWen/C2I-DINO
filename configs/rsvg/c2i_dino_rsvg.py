_base_ = '../_base_/rsvg_class_suffix_base.py'

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
            # Paper setting: exclude queries overlapping any GT by IoU >= 0.7.
            exclude_gt_iou_thr=0.7)))

# Train the BERT language encoder with a 0.1 learning-rate multiplier.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'language_model': dict(lr_mult=0.1)}))

# The RSVG protocol is evaluated at Pr@0.25 and Pr@0.5 (rather than the
# OPT/DIOR 0.5--0.9 sweep inherited from the generic base config).
val_evaluator = dict(iou_thrs=(0.25, 0.5))
test_evaluator = dict(iou_thrs=(0.25, 0.5))

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='refexp/rsvg_val_Pr@0.5',
        rule='greater'),
    logger=dict(interval=50))

work_dir = 'work_dirs/rsvg'
