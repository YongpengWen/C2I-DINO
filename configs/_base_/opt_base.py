import os

_base_ = './grounding_swin/grounding_dino_swin-t_pretrain_obj365.py'

lang_model_name = 'pretrained/bert-base-uncased'

dataset_type = 'ODVGDataset'
data_root = os.getenv('OPT_DATA_ROOT', 'datasets/opt-rsvg')

load_from = 'pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa

model = dict(
    language_model=dict(name=lang_model_name),
    backbone=dict(init_cfg=None))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadTextAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='odvg_ann/train_vg.jsonl',
        data_prefix=dict(img='Image/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        return_classes=True,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='odvg_ann/val_vg.jsonl',
        data_prefix=dict(img='Image/'),
        pipeline=test_pipeline,
        return_classes=True,
        test_mode=True,
        backend_args=None))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='odvg_ann/test_vg.jsonl',
        data_prefix=dict(img='Image/'),
        pipeline=test_pipeline,
        return_classes=True,
        test_mode=True,
        backend_args=None))

val_evaluator = dict(
    _delete_=True,
    type='DumpODVGResults',
    outfile_path='work_dirs/opt_vg/val_predictions.jsonl',
    ann_file=os.path.join(data_root, 'odvg_ann/val_vg.jsonl'),
    img_prefix=os.path.join(data_root, 'Image') + os.sep,
    score_thr=0.1,
    nms_thr=0.5)

test_evaluator = dict(
    _delete_=True,
    type='DumpODVGResults',
    outfile_path='work_dirs/opt_vg/test_predictions.jsonl',
    ann_file=os.path.join(data_root, 'odvg_ann/test_vg.jsonl'),
    img_prefix=os.path.join(data_root, 'Image') + os.sep,
    score_thr=0.1,
    nms_thr=0.5)

max_epochs = 10
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
        milestones=[8, 9],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3),
    logger=dict(interval=50))

auto_scale_lr = dict(base_batch_size=64)
