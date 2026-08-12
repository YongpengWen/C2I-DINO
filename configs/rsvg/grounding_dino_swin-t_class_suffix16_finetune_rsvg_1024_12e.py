_base_ = './grounding_dino_swin-t_class_suffix_finetune_rsvg_1024_12e.py'

# K=16 sensitivity setting. All remaining data, optimization, and seed choices
# match the K=4 and K=8 CSP experiments.
model = dict(suffix_len=16)

work_dir = '/root/autodl-tmp/work_dirs/rsvg_swin_t_class_suffix16_1024_bs8_12e'
randomness = dict(seed=42, deterministic=False)
