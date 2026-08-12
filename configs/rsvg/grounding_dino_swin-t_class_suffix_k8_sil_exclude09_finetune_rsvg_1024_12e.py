_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_phrase_score_max_finetune_rsvg_1024_12e.py'

model = dict(
    suffix_len=8,
    bbox_head=dict(
        spatial_contrast_cfg=dict(exclude_gt_iou_thr=0.9)))

work_dir = '/root/autodl-tmp/work_dirs/rsvg_cfp_sil_k8_exclude09_seed42'
randomness = dict(seed=42, deterministic=False)
