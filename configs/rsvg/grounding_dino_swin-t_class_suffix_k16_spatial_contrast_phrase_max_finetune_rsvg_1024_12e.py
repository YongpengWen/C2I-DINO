_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_phrase_score_max_finetune_rsvg_1024_12e.py'

model = dict(suffix_len=16)
work_dir = '/root/autodl-tmp/work_dirs/rsvg_cfp_sil_k16_phrase_max_seed42'
randomness = dict(seed=42, deterministic=False)
