_base_ = './grounding_dino_swin-t_finetune_dior_rsvg_trainval_test640_val4.py'

max_epochs = 1
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000)
]

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best=None),
    logger=dict(interval=50))

work_dir = '/root/autodl-tmp/work_dirs/dior_rsvg_trainval_test640_1e'
