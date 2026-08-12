_base_ = './grounding_dino_swin-t_finetune_flir_rgb_224_12e.py'

default_hooks = dict(
    visualization=dict(type='GroundingVisualizationHook', draw=False))
