import argparse
import json
import re
from pathlib import Path


RSVG_CLASSES = (
    'ground track field',
    'basketball court',
    'baseball field',
    'football field',
    'swimming pool',
    'storage tank',
    'water tower',
    'wastewater plant',
    'parking lot',
    'tennis court',
    'roundabout',
    'building',
    'bridge',
    'river',
    'lake',
    'pond',
    'road',
    'tree',
)

COMMAND_PREFIX_RE = re.compile(
    r'^\s*(?:find|search|locate|identify)\s+', re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rewrite RSVG tokens_positive from full sentence to target '
        'noun phrase spans.')
    parser.add_argument(
        '--data-root',
        default='datasets/RSVG/rsvg',
        help='RSVG root that contains odvg_ann and mdetr_annotations.')
    parser.add_argument(
        '--out-suffix',
        default='_processed',
        help='Suffix for generated annotation directories.')
    return parser.parse_args()


def find_target_span(caption):
    lower = caption.lower()
    matches = []
    for cls_name in RSVG_CLASSES:
        pattern = r'(?<![a-z])' + re.escape(cls_name) + r's?(?![a-z])'
        match = re.search(pattern, lower)
        if match:
            matches.append((match.start(), -len(match.group(0)), match.end()))

    if not matches:
        raise ValueError(f'No known RSVG target class found: {caption}')

    cls_start, _, cls_end = min(matches)
    prefix = caption[:cls_start]
    command = COMMAND_PREFIX_RE.match(prefix)
    start = command.end() if command else 0

    while start < cls_start and caption[start].isspace():
        start += 1
    while cls_end < len(caption) and caption[cls_end].isspace():
        cls_end += 1
    return start, cls_end


def rewrite_odvg(in_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    total = 0
    with in_path.open('r', encoding='utf-8') as src, out_path.open(
            'w', encoding='utf-8') as dst:
        for line in src:
            item = json.loads(line)
            caption = item['grounding']['caption']
            start, end = find_target_span(caption)
            for region in item['grounding']['regions']:
                old = region.get('tokens_positive')
                new = [[start, end]]
                region['tokens_positive'] = new
                region['phrase'] = caption[start:end]
                changed += old != new
                total += 1
            dst.write(json.dumps(item, ensure_ascii=False) + '\n')
    return total, changed


def rewrite_mdetr(in_path, out_path):
    data = json.loads(in_path.read_text(encoding='utf-8'))
    image_by_id = {image['id']: image for image in data['images']}
    changed = 0
    total = 0
    for ann in data['annotations']:
        caption = image_by_id[ann['image_id']]['caption']
        start, end = find_target_span(caption)
        old = ann.get('tokens_positive')
        new = [[start, end]]
        ann['tokens_positive'] = new
        changed += old != new
        total += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return total, changed


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    odvg_in = data_root / 'odvg_ann'
    odvg_out = data_root / f'odvg_ann{args.out_suffix}'
    for split in ('train', 'val', 'test'):
        total, changed = rewrite_odvg(
            odvg_in / f'{split}_vg.jsonl',
            odvg_out / f'{split}_vg.jsonl')
        print(f'ODVG {split}: regions={total}, changed={changed}')

    mdetr_in = data_root / 'mdetr_annotations'
    mdetr_out = data_root / f'mdetr_annotations{args.out_suffix}'
    for split in ('train', 'val', 'test'):
        total, changed = rewrite_mdetr(
            mdetr_in / f'finetune_rsvg_{split}.json',
            mdetr_out / f'finetune_rsvg_{split}.json')
        print(f'MDETR {split}: annotations={total}, changed={changed}')


if __name__ == '__main__':
    main()
