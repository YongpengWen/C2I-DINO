_base_ = './grounding_dino_swin-t_class_suffix_finetune_rsvg_1024_12e.py'

# K=4 sensitivity setting. All remaining data, optimization, and seed choices
# match the K=8 CSP experiment.
model = dict(suffix_len=4)

work_dir = '/root/autodl-tmp/work_dirs/rsvg_swin_t_class_suffix4_1024_bs8_12e'
randomness = dict(seed=42, deterministic=False)
