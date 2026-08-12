# OPT Visual Grounding

This folder contains the OPT visual grounding helper scripts. The matching
training config is:

```text
configs/opt_vg/grounding_dino_swin-t_finetune_opt_vg.py
```

## Data

Expected local layout:

```text
/root/autodl-tmp/opt-rsvg
  Image/
  Annotations/
  split/
  odvg_ann/
```

On AutoDL, set:

```bash
export OPT_DATA_ROOT=/root/autodl-tmp/opt-rsvg
```

## Convert

```bash
python tools/opt_vg/opt_vg_to_odvg.py --data-root "$OPT_DATA_ROOT"
```

## Train

```bash
python tools/train.py configs/opt_vg/grounding_dino_swin-t_finetune_opt_vg.py \
  --work-dir /root/autodl-tmp/work_dirs/opt_vg
```

Or use the wrapper:

```bash
bash tools/opt_vg/run_autodl.sh train
```

## Inference

```bash
export WEIGHTS=/root/autodl-tmp/work_dirs/opt_vg/epoch_12.pth
bash tools/opt_vg/run_autodl.sh infer
```
