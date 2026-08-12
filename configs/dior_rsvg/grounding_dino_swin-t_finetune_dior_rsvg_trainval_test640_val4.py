_base_ = './grounding_dino_swin-t_finetune_dior_rsvg_official_split_resize640.py'

custom_imports = dict(
    imports=['mmdet.evaluation.metrics.refexp_metric'],
    allow_failed_imports=False)

# Experiment protocol:
# - train on DIOR-RSVG train+val
# - validate/report on DIOR-RSVG test
# - keep inference size fixed at 640x640
# - evaluate once after the final epoch
train_dataloader = dict(
    dataset=dict(ann_file='odvg_ann_official_split/trainval_vg.jsonl'))

val_dataloader = dict(
    dataset=dict(
        ann_file='mdetr_annotations_official_split/finetune_dior_rsvg_test.json'))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiBoxRefExpMetric',
    ann_file=_base_.mdetr_ann_root + 'finetune_dior_rsvg_test.json',
    metric='bbox',
    iou_thrs=(0.5, 0.6, 0.7, 0.8, 0.9),
    dataset_name='dior_rsvg_test')
test_evaluator = val_evaluator

max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=12)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.5)
]

default_hooks = dict(
    checkpoint=dict(
        interval=12,
        max_keep_ckpts=3,
        save_best=None),
    logger=dict(interval=50))

work_dir = '/root/autodl-tmp/work_dirs/dior_rsvg_swin_t_12e'
