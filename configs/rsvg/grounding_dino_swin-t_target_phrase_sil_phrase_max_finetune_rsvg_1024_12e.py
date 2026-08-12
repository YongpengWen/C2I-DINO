_base_ = './grounding_dino_swin-t_target_phrase_finetune_rsvg_1024_12e.py'

custom_imports = dict(
    imports=[
        'mmdet.evaluation.metrics.refexp_metric',
        'mmdet.models.dense_heads.grounding_dino_spatial_contrast_head',
    ],
    allow_failed_imports=False)

# SIL-only reference for the prompt-length sweep.
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
            exclude_gt_iou_thr=0.5)))

work_dir = '/root/autodl-tmp/work_dirs/rsvg_sil_k0_phrase_max_seed42'
randomness = dict(seed=42, deterministic=False)
