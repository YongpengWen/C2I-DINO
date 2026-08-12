_base_ = './grounding_dino_swin-t_finetune_flir_ir_224_12e.py'

image_prefix = 'image_data/flir/rgb/'

train_dataloader = dict(dataset=dict(data_prefix=dict(img=image_prefix)))
val_dataloader = dict(dataset=dict(data_prefix=dict(img=image_prefix)))
test_dataloader = dict(dataset=dict(data_prefix=dict(img=image_prefix)))

work_dir = '/root/autodl-tmp/work_dirs/flir_rgb_gdino_swin_t_224_12e'
