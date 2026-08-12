_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py'

model = dict(suffix_len=16)
test_dataloader = dict(
    dataset=dict(ann_file='mdetr_annotations_target_np/finetune_rsvg_val.json'))
test_evaluator = dict(
    ann_file='/root/autodl-tmp/datasets/RSVG/rsvg/mdetr_annotations/finetune_rsvg_val.json',
    dataset_name='rsvg_val',
    iou_thrs=(0.25, 0.5))
