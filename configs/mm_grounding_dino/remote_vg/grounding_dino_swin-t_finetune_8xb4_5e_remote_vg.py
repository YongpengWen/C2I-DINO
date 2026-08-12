_base_ = '../refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcoco.py'

data_root = 'data/remote_vg/'
lang_model_name = '/root/autodl-tmp/pretrained/bert-base-uncased'
work_dir = 'work_dirs/remote_vg'

model = dict(
    language_model=dict(name=lang_model_name),
    backbone=dict(init_cfg=None))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
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
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations/train_vg.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline,
        backend_args=None))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations/val_vg.json',
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline,
        return_classes=True,
        test_mode=True,
        backend_args=None))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations/test_vg.json',
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline,
        return_classes=True,
        test_mode=True,
        backend_args=None))

val_evaluator = dict(
    _delete_=True,
    type='DumpODVGResults',
    outfile_path='work_dirs/remote_vg/val_predictions.jsonl',
    img_prefix=data_root + 'images/',
    score_thr=0.1,
    nms_thr=0.5)

test_evaluator = dict(
    _delete_=True,
    type='DumpODVGResults',
    outfile_path='work_dirs/remote_vg/test_predictions.jsonl',
    img_prefix=data_root + 'images/',
    score_thr=0.1,
    nms_thr=0.5)

max_epochs = 20
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16],
        gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))

load_from = '/root/autodl-tmp/pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
