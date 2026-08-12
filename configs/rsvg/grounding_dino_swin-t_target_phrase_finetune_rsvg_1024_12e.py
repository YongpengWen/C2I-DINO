_base_ = './grounding_dino_swin-t_finetune_rsvg_1024_12e.py'

# Target Phrase Only ablation:
# - keep the original GroundingDINO model and all training hyperparameters
# - only replace annotations with target-phrase tokens_positive versions
# - do not import or enable class-specific suffix modules

mdetr_ann_root = _base_.data_root + '/mdetr_annotations_target_np/'

train_dataloader = dict(
    dataset=dict(ann_file='odvg_ann_target_np/train_vg.jsonl'))

val_dataloader = dict(
    dataset=dict(ann_file='mdetr_annotations_target_np/finetune_rsvg_val.json'))

test_dataloader = dict(
    dataset=dict(
        ann_file='mdetr_annotations_target_np/finetune_rsvg_test.json'))

val_evaluator = dict(
    ann_file=mdetr_ann_root + 'finetune_rsvg_val.json')

test_evaluator = dict(
    ann_file=mdetr_ann_root + 'finetune_rsvg_test.json')

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'rsvg_swin_t_target_phrase_1024_bs8_12e')

randomness = dict(seed=42, deterministic=False)
