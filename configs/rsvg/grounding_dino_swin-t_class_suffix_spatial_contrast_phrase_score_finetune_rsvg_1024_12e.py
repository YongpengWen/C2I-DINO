_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py'

model = dict(
    bbox_head=dict(
        spatial_contrast_cfg=dict(phrase_score_aggregation='logsumexp')))

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'rsvg_cfp_sil_phrase_logsumexp_seed42')

randomness = dict(seed=42, deterministic=False)
