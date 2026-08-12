"""Render difficult RSVG test cases as GT, baseline, and proposed triplets."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RELATION_PATTERN = re.compile(
    r'\b(north|south|east|west|upper|lower|left|right|near|next to|'
    r'between|adjacent|around|beside|corner|middle|center|first|second|'
    r'third|largest|smallest)\b', re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--baseline-results', required=True)
    parser.add_argument('--ours-results', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--hide-text', action='store_true')
    parser.add_argument('--indices-file', default=None)
    parser.add_argument('--manifest-name', default='selection.json')
    return parser.parse_args()


def font(size):
    path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(path, size=size) if Path(path).exists() else ImageFont.load_default()


def resize(image, boxes, max_side=1400):
    scale = min(1.0, max_side / max(image.size))
    if scale == 1.0:
        return image, boxes
    image = image.resize((round(image.width * scale), round(image.height * scale)))
    return image, [[value * scale for value in box] for box in boxes]


def draw_box(draw, box, color, width=4):
    draw.rectangle([round(value) for value in box], outline=color, width=width)


def draw_text(draw, xy, text, color, text_font):
    x, y = xy
    bounds = draw.textbbox((x, y), text, font=text_font)
    draw.rectangle((bounds[0] - 4, bounds[1] - 3, bounds[2] + 4, bounds[3] + 3), fill='white')
    draw.text((x, y), text, font=text_font, fill=color)


def render(item, mode, out_path, hide_text=False):
    image = Image.open(item['image_path']).convert('RGB')
    boxes = item['gt_boxes'] if mode == 'gt' else [item['prediction']['bbox']]
    image, boxes = resize(image, boxes)
    draw = ImageDraw.Draw(image)
    title_font, label_font = font(18), font(15)
    if mode == 'gt':
        for box in boxes:
            draw_box(draw, box, (28, 150, 76))
        title = f"GT | sample {item['dataset_index']}"
        color = (28, 150, 76)
    else:
        draw_box(draw, boxes[0], (220, 53, 69))
        score = item['prediction']['score']
        title = f"{mode.capitalize()} | IoU={item[f'{mode}_iou']:.3f} | score={score:.3f}"
        color = (220, 53, 69)
    if not hide_text:
        draw_text(draw, (10, 10), title, color, title_font)
        draw_text(draw, (10, 42), item['caption'][:180], (20, 24, 28), label_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main():
    args = parse_args()
    annotations = json.loads(Path(args.ann_file).read_text())
    baseline = {entry['dataset_index']: entry for entry in json.loads(Path(args.baseline_results).read_text())}
    ours = {entry['dataset_index']: entry for entry in json.loads(Path(args.ours_results).read_text())}
    by_file = defaultdict(list)
    for index, image in enumerate(annotations['images']):
        by_file[image['file_name']].append(image)

    candidates = []
    for index, image in enumerate(annotations['images']):
        if index not in baseline or index not in ours:
            continue
        peers = by_file[image['file_name']]
        repeated_instances = sum(peer.get('target_class') == image.get('target_class') for peer in peers)
        if repeated_instances < 2:
            continue
        if not args.indices_file and not RELATION_PATTERN.search(image['caption']):
            continue
        base, proposed = baseline[index], ours[index]
        candidates.append(dict(
            dataset_index=index,
            image_id=image['id'],
            file_name=image['file_name'],
            caption=image['caption'],
            target_class=image.get('target_class'),
            repeated_instances=repeated_instances,
            image_path=proposed['image_path'],
            gt_boxes=proposed['gt_boxes'],
            baseline_iou=base['top1_iou'],
            ours_iou=proposed['top1_iou'],
            baseline_prediction=base['predictions'][0],
            ours_prediction=proposed['predictions'][0],
        ))

    if args.indices_file:
        indices = json.loads(Path(args.indices_file).read_text())
        by_index = {item['dataset_index']: item for item in candidates}
        selected = [by_index[index] for index in indices if index in by_index]
    else:
        selected = [item for item in candidates if item['baseline_iou'] < 0.5 and item['ours_iou'] >= 0.9]
        if len(selected) < args.count:
            selected = [item for item in candidates if item['baseline_iou'] < 0.5 and item['ours_iou'] >= 0.8]
        if len(selected) < args.count:
            selected = [item for item in candidates if item['baseline_iou'] < 0.5 and item['ours_iou'] >= 0.7]
        selected.sort(key=lambda item: (item['baseline_iou'] - item['ours_iou'], -item['ours_iou']))
        unique_selected = []
        seen_files = set()
        for item in selected:
            if item['file_name'] in seen_files:
                continue
            unique_selected.append(item)
            seen_files.add(item['file_name'])
            if len(unique_selected) == args.count:
                break
        selected = unique_selected
    if len(selected) < args.count:
        raise RuntimeError(f'Only found {len(selected)} difficult baseline-failure cases.')

    output = Path(args.out_dir)
    manifest = []
    for rank, item in enumerate(selected, start=1):
        stem = f'{rank:02d}_{Path(item["file_name"]).stem}'
        render(item, 'gt', output / 'GT' / f'{stem}.png', args.hide_text)
        item['prediction'] = item.pop('baseline_prediction')
        render(item, 'baseline', output / 'baseline' / f'{stem}.png', args.hide_text)
        item['prediction'] = item.pop('ours_prediction')
        render(item, 'ours', output / 'ours' / f'{stem}.png', args.hide_text)
        manifest.append(item)
    (output / args.manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f'Rendered {len(selected)} cases to {output}')


if __name__ == '__main__':
    main()
