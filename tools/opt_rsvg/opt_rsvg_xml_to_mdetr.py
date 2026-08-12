import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert official OPT-RSVG XML annotations and split txt '
        'to MDETR/RefCOCO-style COCO json.')
    parser.add_argument(
        '--data-root',
        default='datasets/opt-rsvg',
        help='OPT-RSVG root containing Annotations, split and Image.')
    parser.add_argument(
        '--out-dir',
        default='mdetr_annotations_official_split',
        help='Output directory under DATA_ROOT.')
    return parser.parse_args()


def iter_xml_objects(ann_dir):
    xml_paths = []
    for dirpath, _, filenames in os.walk(ann_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_paths.append(Path(dirpath) / filename)

    # The official split files index the flattened object list. Sorting makes
    # the conversion deterministic and matches the numeric filename order.
    for xml_path in sorted(xml_paths):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext('./filename')
        size = root.find('./size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
        for obj in root.findall('object'):
            box = obj.find('bndbox')
            phrase = obj.findtext('description') or obj.findtext('name') or ''
            yield {
                'filename': filename,
                'width': width,
                'height': height,
                'phrase': phrase.strip(),
                'bbox': [
                    float(box.findtext('xmin')),
                    float(box.findtext('ymin')),
                    float(box.findtext('xmax')),
                    float(box.findtext('ymax')),
                ],
            }


def convert_split(objects, split_path, out_path, dataset_name):
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'object'}]

    with split_path.open('r', encoding='utf-8') as f:
        indices = [int(line.strip()) for line in f if line.strip()]

    for img_id, obj_idx in enumerate(indices, 1):
        obj = objects[obj_idx]
        phrase = obj['phrase']
        images.append({
            'id': img_id,
            'file_name': obj['filename'],
            'height': obj['height'],
            'width': obj['width'],
            'caption': phrase,
            'dataset_name': dataset_name,
            'source_object_idx': obj_idx,
        })

        x1, y1, x2, y2 = obj['bbox']
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        annotations.append({
            'id': img_id,
            'image_id': img_id,
            'category_id': 1,
            'bbox': [x1, y1, w, h],
            'area': w * h,
            'iscrowd': 0,
            'tokens_positive': [[0, len(phrase)]],
        })

    out = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding='utf-8')
    print(
        f'{split_path.name}: images={len(images)}, '
        f'annotations={len(annotations)}, out={out_path}')


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    objects = list(iter_xml_objects(data_root / 'Annotations'))
    print(f'xml objects={len(objects)}')

    out_dir = data_root / args.out_dir
    for split in ['train', 'val', 'test']:
        convert_split(
            objects,
            data_root / 'split' / f'{split}.txt',
            out_dir / f'finetune_opt_rsvg_{split}.json',
            'opt_rsvg')


if __name__ == '__main__':
    main()
