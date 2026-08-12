_base_ = './grounding_dino_swin-t_class_suffix_finetune_rsvg_1024_12e.py'

test_dataloader = _base_.val_dataloader
test_evaluator = _base_.val_evaluator
test_evaluator.iou_thrs = (0.25, 0.5)
