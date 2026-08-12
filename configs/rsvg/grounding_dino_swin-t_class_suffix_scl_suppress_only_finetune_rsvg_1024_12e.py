_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py'

# SCL design ablation: retain the negative-score suppression term only.
model = dict(
    suffix_len=8,
    bbox_head=dict(
        spatial_contrast_cfg=dict(
            rank_weight=0.0,
            suppress_weight=0.5)))

# Use the RSVG reporting protocol consistently during validation and testing.
val_evaluator = dict(iou_thrs=(0.25, 0.5))
test_evaluator = dict(iou_thrs=(0.25, 0.5))

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'rsvg_swin_t_class_suffix_scl_suppress_only_1024_bs8_12e')

randomness = dict(seed=42, deterministic=False)
