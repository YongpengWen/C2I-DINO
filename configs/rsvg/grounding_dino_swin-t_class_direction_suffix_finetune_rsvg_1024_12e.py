_base_ = './grounding_dino_swin-t_class_suffix_finetune_rsvg_1024_12e.py'

rsvg_direction_names = [
    'unknown',
    'left',
    'right',
    'upper',
    'lower',
    'middle',
    'upper_left',
    'upper_right',
    'lower_left',
    'lower_right',
]

model = dict(
    direction_names=rsvg_direction_names,
    direction_suffix_len=4)

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'rsvg_swin_t_class_direction_suffix_1024_bs8_12e')

randomness = dict(seed=42, deterministic=False)
