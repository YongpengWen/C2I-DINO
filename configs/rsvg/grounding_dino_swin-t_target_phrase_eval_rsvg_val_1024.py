_base_ = './grounding_dino_swin-t_target_phrase_finetune_rsvg_1024_12e.py'

# Evaluate the validation split with the same protocol as the GeoVG table.
test_dataloader = _base_.val_dataloader
test_evaluator = _base_.val_evaluator
test_evaluator.iou_thrs = (0.25, 0.5, 0.6, 0.7, 0.8, 0.9)

work_dir = '/root/autodl-tmp/work_dirs/rsvg_geovg_table_gdino_val'
