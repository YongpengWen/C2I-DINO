# DIOR-RSVG GroundingDINO

Expected local layout:

```text
/root/autodl-tmp/DIOR-RSVG
  Annotations/
  JPEGImages/
  train.txt
  val.txt
  test.txt
```

If the split txt files are stored in `split/`, the converter detects that
layout automatically.

Convert annotations:

```bash
export DIOR_DATA_ROOT=/root/autodl-tmp/DIOR-RSVG
python tools/dior_rsvg/dior_rsvg_xml_to_grounding.py --data-root "$DIOR_DATA_ROOT"
```

Train:

```bash
PYTHONPATH=/root/autodl-tmp/mmdetection-dino:$PYTHONPATH \
python tools/train.py configs/dior_rsvg/grounding_dino_swin-t_finetune_dior_rsvg_official_split.py \
  --work-dir /root/autodl-tmp/work_dirs/dior_rsvg_official_split_bs8_12e
```

Train with the TCM matching cost from Efficient Grounding DINO:

```bash
PYTHONPATH=/root/autodl-tmp/mmdetection-dino:$PYTHONPATH \
python tools/train.py configs/dior_rsvg/grounding_dino_swin-t_tcm_finetune_dior_rsvg_official_split.py \
  --work-dir /root/autodl-tmp/work_dirs/dior_rsvg_tcm_official_split_bs8_12e
```

Test:

```bash
PYTHONPATH=/root/autodl-tmp/mmdetection-dino:$PYTHONPATH \
python tools/test.py configs/dior_rsvg/grounding_dino_swin-t_finetune_dior_rsvg_official_split.py \
  /root/autodl-tmp/work_dirs/dior_rsvg_official_split_bs8_12e/best_refexp_dior_rsvg_val_Pr@0.5_epoch_*.pth \
  --work-dir /root/autodl-tmp/work_dirs/dior_rsvg_official_split_bs8_12e/test_best
```

If your image directory is named `Image` instead of `JPEGImages`, set:

```bash
export DIOR_IMAGE_DIR=Image
```
