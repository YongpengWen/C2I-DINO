_base_ = './grounding_dino_swin-t_class_suffix_only_finetune_rsvg_1024_12e.py'

# Target-NP supervision with class-specific suffix tokens only; no SIL loss.
model = dict(suffix_len=16)

work_dir = '/root/autodl-tmp/work_dirs/rsvg_cfp_k16_class_only_seed42'
randomness = dict(seed=42, deterministic=False)
