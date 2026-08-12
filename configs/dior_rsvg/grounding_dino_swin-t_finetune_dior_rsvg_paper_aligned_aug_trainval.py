_base_ = './grounding_dino_swin-t_finetune_dior_rsvg_paper_aligned_640_trainval.py'

# Paper "*" protocol: use DIOR-RSVG scale augmentation for training.
# Inference remains 640x640.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 480), (560, 560), (640, 640), (720, 720), (800, 800)],
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = '/root/autodl-tmp/work_dirs/dior_rsvg_paper_aligned_aug_trainval_bs16_12e'
