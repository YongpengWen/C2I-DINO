_base_ = './grounding_dino_swin-t_class_suffix_k8_sil_exclude09_finetune_rsvg_1024_12e.py'

test_dataloader = _base_.val_dataloader
test_evaluator = dict(
    ann_file='/root/autodl-tmp/datasets/RSVG/rsvg/mdetr_annotations/finetune_rsvg_val.json',
    iou_thrs=(0.25, 0.5))
