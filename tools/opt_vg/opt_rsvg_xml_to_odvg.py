import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert official OPT-RSVG XML annotations and split txt '
        'to one-object-per-line ODVG jsonl files.')
    parser.add_argument(
        '--data-root',
        default='/root/autodl-tmp/opt-rsvg',
        help='OPT-RSVG root containing Annotations, split and Image.')
    parser.add_argument(
        '--out-dir',
        default='odvg_ann_official_split',
        help='Output directory under DATA_ROOT.')
    return parser.parse_args()


def iter_xml_objects(ann_dir):
    xml_paths = []
    for dirpath, _, filenames in os.walk(ann_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_paths.append(Path(dirpath) / filename)

    for xml_path in sorted(xml_paths):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext('./filename')
        size = root.find('./size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))

        for obj in root.findall('object'):
            box = obj.find('bndbox')
            phrase = obj.findtext('description') or obj.findtext('name') or ''
            phrase = phrase.strip()
            yield {
                'filename': filename,
                'height': height,
                'width': width,
                'phrase': phrase,
                'bbox': [
                    float(box.findtext('xmin')),
                    float(box.findtext('ymin')),
                    float(box.findtext('xmax')),
                    float(box.findtext('ymax')),
                ],
            }


def convert_split(objects, split_path, out_path, skip_invalid=False):
    with split_path.open('r', encoding='utf-8') as f:
        indices = [int(line.strip()) for line in f if line.strip()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with out_path.open('w', encoding='utf-8') as f:
        for obj_idx in indices:
            obj = objects[obj_idx]
            phrase = obj['phrase']
            x1, y1, x2, y2 = obj['bbox']
            if skip_invalid and (x2 - x1 < 1 or y2 - y1 < 1):
                skipped += 1
                continue
            item = {
                'filename': obj['filename'],
                'height': obj['height'],
                'width': obj['width'],
                'grounding': {
                    'caption': phrase,
                    'regions': [{
                        'bbox': obj['bbox'],
                        'phrase': phrase,
                        'tokens_positive': [[0, len(phrase)]],
                    }],
                },
            }
            f.write(json.dumps(item) + '\n')

    print(
        f'{split_path.name}: lines={len(indices) - skipped}, '
        f'skipped={skipped}, out={out_path}')


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
            out_dir / f'{split}_vg.jsonl',
            skip_invalid=(split == 'train'))


if __name__ == '__main__':
    main()
