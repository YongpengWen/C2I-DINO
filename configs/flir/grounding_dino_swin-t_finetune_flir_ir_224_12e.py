_base_ = './grounding_dino_swin-t_finetune_flir_ir_640_12e.py'

# Match the 224 x 224 input resolution reported by MM-VG for FLIR.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='RandomChoiceResize', scales=[(224, 224)], keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPos', tokenizer_name=_base_.lang_model_name,
         num_sample_negative=85, max_tokens=256),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip',
        'flip_direction', 'text', 'custom_entities', 'tokens_positive',
        'dataset_mode')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(224, 224), keep_ratio=True,
         backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadTextAnnotations'),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text',
        'custom_entities', 'tokens_positive')),
]

train_dataloader = dict(batch_size=16, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

model = dict(num_queries=100, test_cfg=dict(max_per_img=100))

optim_wrapper = dict(optimizer=dict(lr=0.0002))

work_dir = '/root/autodl-tmp/work_dirs/flir_ir_gdino_swin_t_224_12e'
