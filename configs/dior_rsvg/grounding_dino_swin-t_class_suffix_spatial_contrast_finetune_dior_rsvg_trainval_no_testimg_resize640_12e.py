_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_dior_rsvg_official_split_resize640_12e.py'

train_dataloader = dict(
    dataset=dict(
        ann_file='odvg_ann_official_split_target_np/'
        'trainval_no_testimg_vg.jsonl'))

val_dataloader = dict(
    dataset=dict(
        ann_file='mdetr_annotations_official_split_target_np/'
        'finetune_dior_rsvg_test.json'))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file='/root/autodl-tmp/datasets/DIOR-RSVG/'
    'mdetr_annotations_official_split/finetune_dior_rsvg_test.json',
    dataset_name='dior_rsvg_test')

test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best='refexp/dior_rsvg_test_Pr@0.5'))

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'dior_rsvg_swin_t_class_suffix_spatial_contrast_'
    'trainval_no_testimg_resize640_12e')
