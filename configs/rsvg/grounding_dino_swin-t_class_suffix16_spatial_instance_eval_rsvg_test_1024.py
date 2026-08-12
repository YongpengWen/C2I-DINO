_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py'

model = dict(suffix_len=16)
test_evaluator = dict(iou_thrs=(0.25, 0.5))
