_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py'

# Keep the original max token aggregation for each target phrase.
work_dir = '/root/autodl-tmp/work_dirs/rsvg_cfp_sil_phrase_max_seed42'

randomness = dict(seed=42, deterministic=False)
