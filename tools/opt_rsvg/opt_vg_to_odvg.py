import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OPT visual grounding XML annotations to ODVG jsonl.')
    parser.add_argument(
        '--data-root',
        default='datasets/opt-rsvg',
        help='OPT dataset root containing Annotations, Image and split.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory. Defaults to DATA_ROOT/odvg_ann.')
    return parser.parse_args()


def normalize_image_id(raw_id):
    raw_id = raw_id.strip()
    stem = Path(raw_id).stem
    if stem.isdigit():
        return f'{int(stem):06d}'
    return stem


def find_text_span(caption, phrase, search_from=0):
    start = caption.find(phrase, search_from)
    if start < 0:
        start = caption.find(phrase)
    if start < 0:
        raise ValueError(f'Phrase not found in caption: {phrase}')
    return start, start + len(phrase)


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext('filename')
    width = int(float(root.findtext('size/width')))
    height = int(float(root.findtext('size/height')))

    grouped_regions = {}
    phrase_order = []
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        if box is None:
            continue

        phrase = obj.findtext('description') or obj.findtext('name')
        if phrase is None:
            continue
        phrase = ' '.join(phrase.strip().split())
        if not phrase:
            continue

        xmin = float(box.findtext('xmin'))
        ymin = float(box.findtext('ymin'))
        xmax = float(box.findtext('xmax'))
        ymax = float(box.findtext('ymax'))
        bbox = [
            max(0.0, xmin),
            max(0.0, ymin),
            min(float(width), xmax),
            min(float(height), ymax),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        if phrase not in grouped_regions:
            grouped_regions[phrase] = []
            phrase_order.append(phrase)
        grouped_regions[phrase].append(bbox)

    caption_parts = [phrase.rstrip('.') for phrase in phrase_order]
    caption = '. '.join(caption_parts)
    if caption and not caption.endswith('.'):
        caption += '.'

    regions = []
    search_from = 0
    for phrase in phrase_order:
        phrase_in_caption = phrase.rstrip('.')
        start, end = find_text_span(caption, phrase_in_caption, search_from)
        search_from = end
        bboxes = grouped_regions[phrase]
        regions.append({
            'bbox': bboxes[0] if len(bboxes) == 1 else bboxes,
            'phrase': phrase_in_caption,
            'tokens_positive': [[start, end]],
        })

    return {
        'filename': filename,
        'height': height,
        'width': width,
        'grounding': {
            'caption': caption,
            'regions': regions,
        },
    }


def convert_split(data_root, split_name, out_path):
    split_file = data_root / 'split' / f'{split_name}.txt'
    ann_dir = data_root / 'Annotations'

    converted = 0
    skipped = 0
    with split_file.open('r', encoding='utf-8') as f, out_path.open(
            'w', encoding='utf-8') as out:
        for line in f:
            image_id = normalize_image_id(line)
            if not image_id:
                continue
            xml_path = ann_dir / f'{image_id}.xml'
            if not xml_path.exists():
                skipped += 1
                continue

            item = parse_xml(xml_path)
            if not item['grounding']['regions']:
                skipped += 1
                continue
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
            converted += 1

    print(f'{split_name}: converted={converted}, skipped={skipped}, out={out_path}')


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / 'odvg_ann'
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        convert_split(data_root, split_name, out_dir / f'{split_name}_vg.jsonl')


if __name__ == '__main__':
    main()
