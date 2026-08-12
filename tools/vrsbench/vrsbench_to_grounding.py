#!/usr/bin/env python3
"""Convert VRSBench referring annotations for this Grounding DINO setup.

The training output is ODVG JSONL, consumed by ``ODVGDataset``.  The official
referring validation annotations are exported as MDETR-style COCO JSON so the
existing ``MDETRStyleRefCocoDataset`` and ``MultiBoxRefExpMetric`` can be used.
"""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=None)
    return parser.parse_args()


def image_sizes(image_dir):
    sizes = {}
    for path in sorted(image_dir.glob('*')):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        with Image.open(path) as image:
            sizes[path.name] = image.size
    return sizes


def hbb_from_corners(corners, width, height):
    if len(corners) != 8:
        raise ValueError(f'Expected 8 corner coordinates, got {corners}')
    xs = corners[0::2]
    ys = corners[1::2]
    x1 = max(0.0, min(xs) * width)
    y1 = max(0.0, min(ys) * height)
    x2 = min(float(width), max(xs) * width)
    y2 = min(float(height), max(ys) * height)
    return [x1, y1, x2, y2]


def phrase_spans(phrase):
    return [[0, len(phrase)]]


def convert_train(root, out_path, sizes):
    annotation_zip = root / 'Annotations_train.zip'
    class_counts = Counter()
    written = skipped_flag = skipped_invalid = missing_images = 0

    with zipfile.ZipFile(annotation_zip) as archive, out_path.open(
            'w', encoding='utf-8') as output:
        for member in sorted(archive.namelist()):
            if not member.startswith('Annotations_train/') or not member.endswith('.json'):
                continue
            annotation = json.loads(archive.read(member))
            filename = annotation['image']
            if filename not in sizes:
                missing_images += 1
                continue
            width, height = sizes[filename]
            for obj in annotation.get('objects', []):
                if obj.get('flag', False):
                    skipped_flag += 1
                    continue
                phrase = obj.get('referring_sentence', '').strip()
                target_class = obj.get('obj_cls', '').strip()
                if not phrase or not target_class:
                    skipped_invalid += 1
                    continue
                try:
                    bbox = hbb_from_corners(obj['obj_corner'], width, height)
                except (KeyError, ValueError):
                    skipped_invalid += 1
                    continue
                if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
                    skipped_invalid += 1
                    continue
                spans = phrase_spans(phrase)
                record = {
                    'filename': filename,
                    'height': height,
                    'width': width,
                    'grounding': {
                        'caption': phrase,
                        'target_class': target_class,
                        'caption_suffix': target_class,
                        'regions': [{
                            'bbox': bbox,
                            'phrase': phrase,
                            'tokens_positive': spans,
                            'target_class': target_class,
                            'target_phrase': phrase,
                            'target_tokens_positive': spans,
                            'caption_suffix': target_class,
                        }],
                    },
                }
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                class_counts[target_class] += 1
                written += 1

    return dict(
        samples=written,
        classes=sorted(class_counts),
        skipped_flag=skipped_flag,
        skipped_invalid=skipped_invalid,
        missing_images=missing_images)


def convert_val(root, out_path, sizes):
    source = json.loads(
        (root / 'VRSBench_EVAL_referring.json').read_text(encoding='utf-8'))
    images, annotations = [], []
    class_counts = Counter()
    skipped_invalid = missing_images = 0

    for image_id, item in enumerate(source, start=1):
        filename = item['image_id']
        if filename not in sizes:
            missing_images += 1
            continue
        width, height = sizes[filename]
        phrase = item['question'].strip()
        target_class = item.get('obj_cls', '').strip()
        try:
            x1, y1, x2, y2 = hbb_from_corners(
                item['obj_corner'], width, height)
        except (KeyError, ValueError):
            skipped_invalid += 1
            continue
        if not phrase or not target_class or x2 - x1 < 1 or y2 - y1 < 1:
            skipped_invalid += 1
            continue
        images.append({
            'id': image_id,
            'file_name': filename,
            'height': height,
            'width': width,
            'caption': phrase,
            'dataset_name': 'vrsbench_referring_val',
            'source_question_id': item['question_id'],
            'target_class': target_class,
            'target_phrase': phrase,
            'caption_suffix': target_class,
        })
        annotations.append({
            'id': len(annotations) + 1,
            'image_id': image_id,
            'category_id': 1,
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'area': (x2 - x1) * (y2 - y1),
            'iscrowd': 0,
            'tokens_positive': phrase_spans(phrase),
            'target_class': target_class,
            'target_phrase': phrase,
            'target_tokens_positive': phrase_spans(phrase),
            'caption_suffix': target_class,
        })
        class_counts[target_class] += 1

    out_path.write_text(json.dumps({
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'object'}],
    }), encoding='utf-8')
    return dict(
        samples=len(images),
        classes=sorted(class_counts),
        skipped_invalid=skipped_invalid,
        missing_images=missing_images)


def main():
    args = parse_args()
    root = args.data_root.resolve()
    out_dir = args.out_dir or root / 'grounding_annotations'
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sizes = image_sizes(root / 'Images_train')
    val_sizes = image_sizes(root / 'Images_val')
    if not train_sizes or not val_sizes:
        raise FileNotFoundError(
            'Images_train/ and Images_val/ must be extracted before conversion.')

    train = convert_train(root, out_dir / 'vrsbench_train_vg.jsonl', train_sizes)
    val = convert_val(root, out_dir / 'vrsbench_referring_val.json', val_sizes)
    classes = sorted(set(train['classes']) | set(val['classes']))
    (out_dir / 'vrsbench_classes.json').write_text(
        json.dumps(classes, indent=2), encoding='utf-8')
    report = {'train': train, 'val': val, 'classes': classes}
    (out_dir / 'conversion_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
