_base_ = './grounding_dino_swin-t_finetune_rsvg_1024_12e.py'

custom_imports = dict(
    imports=[
        'mmdet.evaluation.metrics.refexp_metric',
        'mmdet.models.detectors.grounding_dino_class_suffix',
        'mmdet.models.detectors.grounding_dino_class_suffix_mbores',
    ],
    allow_failed_imports=False)

mdetr_ann_root = _base_.data_root + '/mdetr_annotations/'

rsvg_class_names = [
    'tennis court',
    'roundabout',
    'baseball field',
    'storage tank',
    'bridge',
    'ground track field',
    'football field',
    'basketball court',
    'swimming pool',
    'water tower',
]

model = dict(
    type='GroundingDINOClassSuffixMBORES',
    class_names=rsvg_class_names,
    suffix_len=8,
    suffix_init_std=0.01,
    graph_hidden_dims=256,
    graph_num_heads=8,
    graph_num_layers=2,
    graph_dropout=0.1)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='RandomChoiceResize', scales=[(1024, 1024)], keep_ratio=True),
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
                   'target_class', 'target_phrase', 'caption_suffix'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(1024, 1024),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadTextAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive', 'target_class', 'target_phrase',
                   'caption_suffix'))
]

train_dataloader = dict(
    dataset=dict(
        ann_file='odvg_ann_target_np/train_vg.jsonl',
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        ann_file='mdetr_annotations_target_np/finetune_rsvg_val.json',
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        ann_file='mdetr_annotations_target_np/finetune_rsvg_test.json',
        pipeline=test_pipeline))

val_evaluator = dict(
    ann_file=mdetr_ann_root + 'finetune_rsvg_val.json')

test_evaluator = dict(
    ann_file=mdetr_ann_root + 'finetune_rsvg_test.json')

work_dir = '/root/autodl-tmp/work_dirs/rsvg_swin_t_class_suffix_mbores_1024_bs8_12e'

randomness = dict(seed=42, deterministic=False)
