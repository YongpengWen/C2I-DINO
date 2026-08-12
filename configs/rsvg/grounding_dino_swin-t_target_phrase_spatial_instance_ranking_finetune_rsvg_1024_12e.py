_base_ = './grounding_dino_swin-t_target_phrase_finetune_rsvg_1024_12e.py'

# SIL-only component ablation: retain target-phrase supervision and add only
# the Spatial Instance Learning objective, without Category-Focused Prompting.
custom_imports = dict(
    imports=[
        'mmdet.evaluation.metrics.refexp_metric',
        'mmdet.models.dense_heads.grounding_dino_spatial_contrast_head',
    ],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='GroundingDINOSpatialContrastHead',
        spatial_contrast_cfg=dict(
            enable=True,
            loss_weight=1.0,
            rank_weight=1.0,
            suppress_weight=0.5,
            margin=0.2,
            neg_iou_thr=0.25,
            topk=10,
            score_type='logit',
            exclude_gt_iou_thr=0.7)))

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'rsvg_swin_t_spatial_instance_learning_1024_bs8_12e')

randomness = dict(seed=42, deterministic=False)
