import json
import os

_base_ = '../opt_vg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_opt_vg_official_split_12e.py'

data_root = os.getenv('VRSBENCH_DATA_ROOT', '/root/autodl-fs/VRSBENCH')
ann_root = data_root + '/grounding_annotations/'
with open(ann_root + 'vrsbench_classes.json', encoding='utf-8') as file:
    vrsbench_class_names = json.load(file)
del file

model = dict(class_names=vrsbench_class_names)

lang_model_name = '/root/autodl-tmp/pretrained/bert-base-uncased'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
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
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'target_class', 'target_phrase', 'caption_suffix')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
         backend='pillow'),
    dict(type='LoadTextAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive', 'target_class', 'target_phrase',
                   'caption_suffix')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='grounding_annotations/vrsbench_train_vg.jsonl',
        data_prefix=dict(img='Images_train/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        return_classes=True,
        backend_args=None))

vrsbench_val_dataset = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file='grounding_annotations/vrsbench_referring_val.json',
    data_prefix=dict(img='Images_val/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

vrsbench_val_evaluator = dict(
    type='MultiBoxRefExpMetric',
    ann_file=ann_root + 'vrsbench_referring_val.json',
    metric='bbox',
    iou_thrs=(0.5, 0.6, 0.7, 0.8, 0.9),
    dataset_name='vrsbench_referring_val')

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(_delete_=True, **vrsbench_val_dataset))
test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(_delete_=True, **vrsbench_val_dataset))

val_evaluator = dict(_delete_=True, **vrsbench_val_evaluator)
test_evaluator = dict(_delete_=True, **vrsbench_val_evaluator)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='refexp/vrsbench_referring_val_Pr@0.5',
        rule='greater'),
    logger=dict(interval=50))

work_dir = '/root/autodl-tmp/work_dirs/vrsbench_swin_t_class_suffix_spatial_contrast_12e'
