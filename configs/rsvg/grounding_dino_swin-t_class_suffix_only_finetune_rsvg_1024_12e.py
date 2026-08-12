_base_ = './grounding_dino_swin-t_class_suffix_finetune_rsvg_1024_12e.py'

# Class-specific suffix tokens only; no spatial contrastive head or SIL loss.
work_dir = '/root/autodl-tmp/work_dirs/rsvg_cfp_class_only_seed42'

randomness = dict(seed=42, deterministic=False)
