_base_ = './grounding_dino_swin-t_class_suffix16_only_finetune_rsvg_1024_12e.py'

test_evaluator = dict(iou_thrs=(0.25, 0.5))
