_base_ = './grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_dior_rsvg_trainval_no_testimg_resize640_12e.py'

load_from = '/root/autodl-tmp/work_dirs/dior_rsvg_swin_t_class_suffix_spatial_contrast_trainval_no_testimg_resize640_12e/best_refexp_dior_rsvg_test_Pr@0.5_epoch_11.pth'
resume = False

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
train_dataloader = dict(
    dataset=dict(
        ann_file='odvg_ann_official_split_target_np/trainval_no_testimg_vg_exclude_17679.jsonl'))

optim_wrapper = dict(optimizer=dict(lr=1e-6))
param_scheduler = []
