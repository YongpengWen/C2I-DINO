#!/usr/bin/env python3
"""Convert RGBT-VG FLIR annotations into ODVG and RefCOCO-style files."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    return parser.parse_args()


def load_split(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def to_record(item):
    filename, image_info, xywh, caption = item[:4]
    x, y, width, height = map(float, xywh)
    image_width = int(image_info['width'])
    image_height = int(image_info['height'])
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        'filename': filename,
        'height': image_height,
        'width': image_width,
        'caption': caption,
        'bbox_xyxy': [x1, y1, x2, y2],
    }


def convert_split(records, split, out_root):
    odvg_path = out_root / 'odvg_ann' / f'{split}_vg.jsonl'
    mdetr_path = out_root / 'mdetr_annotations' / f'finetune_flir_{split}.json'
    odvg_path.parent.mkdir(parents=True, exist_ok=True)
    mdetr_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    kept = 0
    with odvg_path.open('w', encoding='utf-8') as f:
        for image_id, item in enumerate(records, start=1):
            record = to_record(item)
            if record is None:
                continue
            caption = record['caption']
            bbox = record['bbox_xyxy']
            f.write(json.dumps({
                'filename': record['filename'],
                'height': record['height'],
                'width': record['width'],
                'grounding': {
                    'caption': caption,
                    'regions': [{
                        'bbox': bbox,
                        'phrase': caption,
                        'tokens_positive': [[0, len(caption)]],
                    }],
                },
            }) + '\n')
            images.append({
                'id': image_id,
                'file_name': record['filename'],
                'height': record['height'],
                'width': record['width'],
                'caption': caption,
                'dataset_name': 'flir',
            })
            annotations.append({
                'id': image_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'iscrowd': 0,
                'tokens_positive': [[0, len(caption)]],
            })
            kept += 1
    mdetr_path.write_text(json.dumps({
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'object'}],
    }), encoding='utf-8')
    print(f'{split}: {kept} samples -> {odvg_path} and {mdetr_path}')


def main():
    args = parse_args()
    ann_root = args.data_root / 'rgbtvg_flir'
    for split in ('train', 'val', 'test', 'testA', 'testB', 'testC'):
        convert_split(load_split(ann_root / f'rgbtvg_flir_{split}.pth'), split,
                      args.data_root)


if __name__ == '__main__':
    main()
