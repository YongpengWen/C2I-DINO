import os

_base_ = './grounding_dino_swin-t_finetune_opt_vg.py'

custom_imports = dict(
    imports=['mmdet.evaluation.metrics.refexp_metric'],
    allow_failed_imports=False)

data_root = os.getenv('OPT_DATA_ROOT', '/root/autodl-tmp/opt-rsvg')
mdetr_ann_root = data_root + '/mdetr_annotations_official_split/'

# Direction words are common in OPT-RSVG. Horizontal flipping changes left and
# right while the text stays unchanged, so disable it for referring training.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='odvg_ann_official_split/train_vg.jsonl',
        data_prefix=dict(img='Image/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        return_classes=True,
        backend_args=None))

val_dataset_opt_rsvg = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file='mdetr_annotations_official_split/finetune_opt_rsvg_val.json',
    data_prefix=dict(img='Image/'),
    test_mode=True,
    return_classes=True,
    pipeline=_base_.test_pipeline,
    backend_args=None)

val_evaluator_opt_rsvg = dict(
    type='MultiBoxRefExpMetric',
    ann_file=mdetr_ann_root + 'finetune_opt_rsvg_val.json',
    metric='bbox',
    iou_thrs=(0.5, 0.6, 0.7, 0.8, 0.9),
    dataset_name='opt_rsvg_val')

test_dataset_opt_rsvg = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file='mdetr_annotations_official_split/finetune_opt_rsvg_test.json',
    data_prefix=dict(img='Image/'),
    test_mode=True,
    return_classes=True,
    pipeline=_base_.test_pipeline,
    backend_args=None)

test_evaluator_opt_rsvg = dict(
    type='MultiBoxRefExpMetric',
    ann_file=mdetr_ann_root + 'finetune_opt_rsvg_test.json',
    metric='bbox',
    iou_thrs=(0.5, 0.6, 0.7, 0.8, 0.9),
    dataset_name='opt_rsvg_test')

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(_delete_=True, **val_dataset_opt_rsvg))

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(_delete_=True, **test_dataset_opt_rsvg))

val_evaluator = dict(_delete_=True, **val_evaluator_opt_rsvg)
test_evaluator = dict(_delete_=True, **test_evaluator_opt_rsvg)

max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
        'language_model': dict(lr_mult=0.1),
    }))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='refexp/opt_rsvg_val_Pr@0.5',
        rule='greater'),
    logger=dict(interval=50))
