"""Convert OPT-RSVG XML annotations to ODVG and MDETR formats."""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OPT-RSVG XML and split files to ODVG JSONL '
        'and MDETR/RefCOCO-style JSON.')
    parser.add_argument('--data-root', default='datasets/opt-rsvg')
    parser.add_argument('--odvg-dir', default='odvg_ann_official_split')
    parser.add_argument('--mdetr-dir', default='mdetr_annotations_official_split')
    return parser.parse_args()


def load_objects(annotation_dir):
    objects = []
    paths = sorted(Path(annotation_dir).rglob('*.xml'))
    for path in paths:
        root = ET.parse(path).getroot()
        size = root.find('./size')
        width = int(float(size.findtext('width')))
        height = int(float(size.findtext('height')))
        filename = root.findtext('./filename')
        for item in root.findall('object'):
            box = item.find('bndbox')
            phrase = (item.findtext('description') or item.findtext('name') or '').strip()
            objects.append({
                'filename': filename,
                'width': width,
                'height': height,
                'phrase': phrase,
                'bbox': [
                    float(box.findtext('xmin')), float(box.findtext('ymin')),
                    float(box.findtext('xmax')), float(box.findtext('ymax')),
                ],
            })
    return objects


def read_indices(path):
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def write_odvg(objects, indices, path, skip_invalid=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open('w', encoding='utf-8') as stream:
        for index in indices:
            obj = objects[index]
            x1, y1, x2, y2 = obj['bbox']
            if skip_invalid and (x2 - x1 < 1 or y2 - y1 < 1):
                continue
            stream.write(json.dumps({
                'filename': obj['filename'],
                'height': obj['height'],
                'width': obj['width'],
                'grounding': {
                    'caption': obj['phrase'],
                    'regions': [{
                        'bbox': obj['bbox'],
                        'phrase': obj['phrase'],
                        'tokens_positive': [[0, len(obj['phrase'])]],
                    }],
                },
            }) + '\n')
            written += 1
    return written


def write_mdetr(objects, indices, path):
    images, annotations = [], []
    for image_id, index in enumerate(indices, 1):
        obj = objects[index]
        x1, y1, x2, y2 = obj['bbox']
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        images.append({
            'id': image_id, 'file_name': obj['filename'],
            'height': obj['height'], 'width': obj['width'],
            'caption': obj['phrase'], 'dataset_name': 'opt_rsvg',
            'source_object_idx': index,
        })
        annotations.append({
            'id': image_id, 'image_id': image_id, 'category_id': 1,
            'bbox': [x1, y1, width, height], 'area': width * height,
            'iscrowd': 0, 'tokens_positive': [[0, len(obj['phrase'])]],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'images': images, 'annotations': annotations,
        'categories': [{'id': 1, 'name': 'object'}],
    }), encoding='utf-8')
    return len(images)


def main():
    args = parse_args()
    root = Path(args.data_root)
    objects = load_objects(root / 'Annotations')
    print(f'XML objects: {len(objects)}')
    for split in ('train', 'val', 'test'):
        indices = read_indices(root / 'split' / f'{split}.txt')
        odvg_count = write_odvg(
            objects, indices, root / args.odvg_dir / f'{split}_vg.jsonl',
            skip_invalid=(split == 'train'))
        mdetr_count = write_mdetr(
            objects, indices, root / args.mdetr_dir / f'finetune_opt_rsvg_{split}.json')
        print(f'{split}: ODVG={odvg_count}, MDETR={mdetr_count}')


if __name__ == '__main__':
    main()
