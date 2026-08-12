_base_ = '../dior_rsvg/grounding_dino_swin-t_finetune_dior_rsvg_official_split_resize640.py'

custom_imports = dict(
    imports=['mmdet.evaluation.metrics.refexp_metric'],
    allow_failed_imports=False)

data_root = '/root/autodl-tmp/datasets/RSVG/rsvg'
image_prefix = 'images/'
mdetr_ann_root = data_root + '/mdetr_annotations/'

train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='odvg_ann/train_vg.jsonl',
        data_prefix=dict(img=image_prefix),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

val_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='MDETRStyleRefCocoDataset',
        data_root=data_root,
        ann_file='mdetr_annotations/finetune_rsvg_val.json',
        data_prefix=dict(img=image_prefix),
        test_mode=True,
        return_classes=True,
        pipeline=_base_.test_pipeline,
        backend_args=None))

test_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='MDETRStyleRefCocoDataset',
        data_root=data_root,
        ann_file='mdetr_annotations/finetune_rsvg_test.json',
        data_prefix=dict(img=image_prefix),
        test_mode=True,
        return_classes=True,
        pipeline=_base_.test_pipeline,
        backend_args=None))

val_evaluator = dict(
    _delete_=True,
    type='MultiBoxRefExpMetric',
    ann_file=mdetr_ann_root + 'finetune_rsvg_val.json',
    metric='bbox',
    iou_thrs=(0.5, 0.6, 0.7, 0.8, 0.9),
    dataset_name='rsvg_val')

test_evaluator = dict(
    _delete_=True,
    type='MultiBoxRefExpMetric',
    ann_file=mdetr_ann_root + 'finetune_rsvg_test.json',
    metric='bbox',
    iou_thrs=(0.5, 0.6, 0.7, 0.8, 0.9),
    dataset_name='rsvg_test')

max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10, 11],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='refexp/rsvg_val_Pr@0.5',
        rule='greater'),
    logger=dict(interval=50))

work_dir = '/root/autodl-tmp/work_dirs/rsvg_swin_t_640_12e'
