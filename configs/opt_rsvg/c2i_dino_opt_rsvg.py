import os

_base_ = '../_base_/opt_split_base.py'

custom_imports = dict(
    imports=[
        'mmdet.evaluation.metrics.refexp_metric',
        'mmdet.models.detectors.grounding_dino_class_suffix',
        'mmdet.models.dense_heads.grounding_dino_spatial_contrast_head',
    ],
    allow_failed_imports=False)

opt_rsvg_class_names = [
    'airplane',
    'ground track field',
    'tennis court',
    'bridge',
    'basketball court',
    'storage tank',
    'ship',
    'baseball diamond',
    't junction',
    'crossroad',
    'parking lot',
    'harbor',
    'vehicle',
    'swimming pool',
]

# The OPT-RSVG files live under the shared datasets directory on this host.
# Keep this explicit so the merged config cannot fall back to the old
# The dataset root defaults to the repository-relative datasets/opt-rsvg path.
data_root = os.getenv('OPT_DATA_ROOT', 'datasets/opt-rsvg')
mdetr_ann_root = data_root + '/mdetr_annotations_official_split/'

model = dict(
    type='GroundingDINOClassSuffix',
    class_names=opt_rsvg_class_names,
    suffix_len=8,
    suffix_init_std=0.01,
    bbox_head=dict(
        type='GroundingDINOSpatialContrastHead',
        spatial_contrast_cfg=dict(
            enable=True,
            loss_weight=1.0,
            margin=0.2,
            neg_iou_thr=0.25,
            topk=10,
            score_type='logit',
            exclude_gt_iou_thr=0.7)))

# Train BERT with a 0.1 learning-rate multiplier.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'language_model': dict(lr_mult=0.1)}))

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
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'target_class', 'target_phrase', 'caption_suffix')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
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
                   'tokens_positive', 'target_class', 'target_phrase',
                   'caption_suffix')),
]

train_dataloader = dict(
    dataset=dict(
        ann_file='odvg_ann_official_split_target_np/train_vg.jsonl',
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        ann_file='mdetr_annotations_official_split_target_np/'
        'finetune_opt_rsvg_val.json',
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        ann_file='mdetr_annotations_official_split_target_np/'
        'finetune_opt_rsvg_test.json',
        pipeline=test_pipeline))

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='refexp/opt_rsvg_val_Pr@0.5',
        rule='greater'),
    logger=dict(interval=50))

work_dir = 'work_dirs/opt_rsvg'
