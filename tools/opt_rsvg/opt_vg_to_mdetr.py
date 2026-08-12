import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OPT-RSVG ODVG jsonl to MDETR/RefCOCO-style COCO json.')
    parser.add_argument(
        '--data-root',
        default='datasets/opt-rsvg',
        help='OPT-RSVG root containing odvg_ann and Image.')
    parser.add_argument(
        '--ann-dir',
        default='odvg_ann',
        help='Directory under DATA_ROOT containing *_vg.jsonl files.')
    parser.add_argument(
        '--out-dir',
        default='mdetr_annotations',
        help='Output directory under DATA_ROOT.')
    return parser.parse_args()


def as_boxes(bbox):
    if not bbox:
        return []
    if isinstance(bbox[0], list):
        return bbox
    return [bbox]


def convert_file(src_path, out_path, dataset_name):
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'object'}]
    img_id = 1
    ann_id = 1

    with src_path.open('r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            for region_idx, region in enumerate(item['grounding']['regions']):
                phrase = region.get('phrase') or item['grounding']['caption']
                if isinstance(phrase, list):
                    phrase = ' '.join(phrase)
                images.append({
                    'id': img_id,
                    'file_name': item['filename'],
                    'height': item['height'],
                    'width': item['width'],
                    'caption': phrase,
                    'dataset_name': dataset_name,
                    'source_filename': item['filename'],
                    'region_idx': region_idx,
                })

                for box in as_boxes(region.get('bbox', [])):
                    x1, y1, x2, y2 = box
                    w = max(0.0, float(x2) - float(x1))
                    h = max(0.0, float(y2) - float(y1))
                    if w <= 0 or h <= 0:
                        continue
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': [float(x1), float(y1), w, h],
                        'area': w * h,
                        'iscrowd': 0,
                        'tokens_positive': [[0, len(phrase)]],
                    })
                    ann_id += 1
                img_id += 1

    out = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding='utf-8')
    print(f'{src_path.name}: images={len(images)}, annotations={len(annotations)}, out={out_path}')


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ann_dir = data_root / args.ann_dir
    out_dir = data_root / args.out_dir
    for split in ['train', 'val', 'test']:
        src_path = ann_dir / f'{split}_vg.jsonl'
        out_path = out_dir / f'finetune_opt_rsvg_{split}.json'
        convert_file(src_path, out_path, 'opt_rsvg')


if __name__ == '__main__':
    main()
