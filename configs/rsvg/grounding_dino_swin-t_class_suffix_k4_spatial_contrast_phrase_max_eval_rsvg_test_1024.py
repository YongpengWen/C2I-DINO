_base_ = './grounding_dino_swin-t_class_suffix_k4_spatial_contrast_phrase_max_finetune_rsvg_1024_12e.py'

test_evaluator = dict(iou_thrs=(0.25, 0.5))
