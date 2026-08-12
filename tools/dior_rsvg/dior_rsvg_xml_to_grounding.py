import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DIOR-RSVG XML annotations and split txt files to '
        'ODVG jsonl for training and MDETR/RefCOCO-style COCO json for '
        'evaluation.')
    parser.add_argument(
        '--data-root',
        default='datasets/DIOR-RSVG',
        help='DIOR-RSVG root containing Annotations, JPEGImages and split txt.')
    parser.add_argument(
        '--ann-dir',
        default='Annotations',
        help='Annotation directory under DATA_ROOT.')
    parser.add_argument(
        '--split-dir',
        default=None,
        help='Split directory under DATA_ROOT. Defaults to '
        'DATA_ROOT/split_official if it exists, then DATA_ROOT/split, '
        'otherwise DATA_ROOT.')
    parser.add_argument(
        '--mdetr-out-dir',
        default='mdetr_annotations_official_split',
        help='MDETR/COCO output directory under DATA_ROOT.')
    parser.add_argument(
        '--odvg-out-dir',
        default='odvg_ann_official_split',
        help='ODVG output directory under DATA_ROOT.')
    parser.add_argument(
        '--one-based-splits',
        action='store_true',
        help='Treat split ids as 1-based instead of 0-based.')
    parser.add_argument(
        '--expected-split-counts',
        nargs=3,
        type=int,
        default=(26991, 3829, 7500),
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Expected train/val/test sample counts. Use 0 0 0 to disable.')
    parser.add_argument(
        '--allow-incomplete-union',
        action='store_true',
        help='Do not require the union of train/val/test ids to cover every '
        'parsed XML object exactly once.')
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
        width = int(float(size.findtext('width')))
        height = int(float(size.findtext('height')))

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


def read_split(split_path, one_based=False):
    with split_path.open('r', encoding='utf-8') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    if one_based:
        indices = [idx - 1 for idx in indices]
    return indices


def resolve_split_dir(data_root, split_dir):
    if split_dir:
        return Path(split_dir)
    for candidate in [data_root / 'split_official', data_root / 'split']:
        if candidate.is_dir():
            return candidate
    return data_root


def load_and_validate_splits(split_dir, one_based, num_objects,
                             expected_split_counts,
                             allow_incomplete_union):
    split_names = ['train', 'val', 'test']
    expected = dict(zip(split_names, expected_split_counts))
    check_counts = any(count > 0 for count in expected_split_counts)
    splits = {}

    for split in split_names:
        split_path = split_dir / f'{split}.txt'
        indices = read_split(split_path, one_based)
        unique_indices = set(indices)

        if len(indices) != len(unique_indices):
            raise ValueError(
                f'{split_path} contains {len(indices) - len(unique_indices)} '
                'duplicate ids.')
        if check_counts and len(indices) != expected[split]:
            raise ValueError(
                f'{split_path} has {len(indices)} ids, expected '
                f'{expected[split]}. This is not the official DIOR-RSVG '
                'split used by the paper.')

        out_of_range = [idx for idx in indices if idx < 0 or idx >= num_objects]
        if out_of_range:
            raise ValueError(
                f'{split_path} contains ids outside [0, {num_objects - 1}], '
                f'for example {out_of_range[:5]}.')

        splits[split] = indices

    for i, split_a in enumerate(split_names):
        ids_a = set(splits[split_a])
        for split_b in split_names[i + 1:]:
            overlap = ids_a & set(splits[split_b])
            if overlap:
                raise ValueError(
                    f'{split_a}/{split_b} split overlap: {len(overlap)} ids, '
                    f'for example {sorted(overlap)[:10]}.')

    union = set().union(*(set(splits[split]) for split in split_names))
    if not allow_incomplete_union and len(union) != num_objects:
        missing = sorted(set(range(num_objects)) - union)
        extra = sorted(union - set(range(num_objects)))
        raise ValueError(
            f'split union has {len(union)} ids but XML has {num_objects} '
            f'objects. Missing examples: {missing[:10]}, '
            f'extra examples: {extra[:10]}.')

    print('validated split counts: ' + ', '.join(
        f'{split}={len(splits[split])}' for split in split_names))
    return splits


def convert_mdetr_split(objects, split_name, indices, out_path, dataset_name):
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'object'}]

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }),
        encoding='utf-8')
    print(
        f'{split_name} mdetr: images={len(images)}, '
        f'annotations={len(annotations)}, out={out_path}')


def convert_odvg_split(objects, split_name, indices, out_path, skip_invalid):
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
        f'{split_name} odvg: lines={len(indices) - skipped}, '
        f'skipped={skipped}, out={out_path}')


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    split_dir = resolve_split_dir(data_root, args.split_dir)
    objects = list(iter_xml_objects(data_root / args.ann_dir))
    print(f'xml objects={len(objects)}')
    print(f'split dir={split_dir}')
    splits = load_and_validate_splits(
        split_dir,
        one_based=args.one_based_splits,
        num_objects=len(objects),
        expected_split_counts=args.expected_split_counts,
        allow_incomplete_union=args.allow_incomplete_union)

    for split in ['train', 'val', 'test']:
        convert_odvg_split(
            objects,
            split,
            splits[split],
            data_root / args.odvg_out_dir / f'{split}_vg.jsonl',
            skip_invalid=(split == 'train'))
        convert_mdetr_split(
            objects,
            split,
            splits[split],
            data_root / args.mdetr_out_dir /
            f'finetune_dior_rsvg_{split}.json',
            dataset_name='dior_rsvg')


if __name__ == '__main__':
    main()
