import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path):
    items = []
    with open(path, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_gt(path):
    gt = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            gt[item['filename']] = item
    return gt


def get_font(size=18):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def color_for_index(idx):
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (233, 196, 106),
        (131, 56, 236),
        (244, 162, 97),
        (38, 70, 83),
        (231, 111, 81),
    ]
    return palette[idx % len(palette)]


def draw_box(draw, box, color, width=4):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)


def iter_boxes(boxes):
    if not boxes:
        return []
    if isinstance(boxes[0], (int, float)):
        return [boxes]
    return boxes


def draw_label(draw, xy, text, fill, font):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    bg = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(bg, fill=(255, 255, 255))
    draw.rectangle(bg, outline=fill, width=2)
    draw.text((x, y), text, fill=fill, font=font)


def render_one(image_path, pred_item, gt_item, out_path, max_regions=6):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = get_font(18)
    small = get_font(15)

    caption = pred_item.get('grounding', {}).get('caption', '')
    draw.rectangle([0, 0, image.width, 34], fill=(255, 255, 255))
    draw.text((8, 8), caption[:180], fill=(0, 0, 0), font=font)

    gt_regions = []
    if gt_item:
        gt_regions = gt_item.get('grounding', {}).get('regions', [])

    pred_regions = pred_item.get('grounding', {}).get('regions', [])[:max_regions]
    y_offset = 42

    for idx, region in enumerate(pred_regions):
        color = color_for_index(idx)
        phrase = region.get('phrase', [])
        phrase_text = phrase[0] if isinstance(phrase, list) and phrase else str(phrase)
        boxes = region.get('bbox', [])
        scores = region.get('score', [[]])
        score_list = scores[0] if scores else []
        for j, box in enumerate(boxes):
            draw_box(draw, box, color, width=3)
            score = score_list[j] if j < len(score_list) else None
            label = phrase_text
            if score is not None:
                label = f'{phrase_text} {score:.2f}'
            draw_label(draw, (8, y_offset), label[:90], color, small)
            y_offset += 20

    if gt_regions:
        for idx, region in enumerate(gt_regions):
            color = (0, 140, 0)
            for box in iter_boxes(region['bbox']):
                draw_box(draw, box, color, width=1)
            label = f"GT: {region.get('phrase', '')}"
            if idx == 0:
                draw_label(draw, (8, y_offset + 4), label[:90], color, small)
                y_offset += 20

    image.save(out_path)


def build_contact_sheet(image_paths, out_path, cols=2, thumb_width=900):
    imgs = [Image.open(p).convert('RGB') for p in image_paths]
    if not imgs:
        return
    thumbs = []
    for img in imgs:
        ratio = thumb_width / img.width
        thumb = img.resize((thumb_width, int(img.height * ratio)))
        thumbs.append(thumb)
    rows = (len(thumbs) + cols - 1) // cols
    row_heights = []
    for r in range(rows):
        row_heights.append(max(thumbs[r * cols + c].height for c in range(min(cols, len(thumbs) - r * cols))))
    canvas = Image.new('RGB', (thumb_width * cols, sum(row_heights)), (245, 245, 245))
    y = 0
    idx = 0
    for r in range(rows):
        x = 0
        h = row_heights[r]
        for c in range(cols):
            if idx >= len(thumbs):
                break
            img = thumbs[idx]
            canvas.paste(img, (x, y))
            x += thumb_width
            idx += 1
        y += h
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='work_dirs/remote_vg/test_predictions.jsonl')
    parser.add_argument('--gt', default='data/remote_vg/annotations/test_vg.json')
    parser.add_argument('--img-root', default='data/remote_vg/images')
    parser.add_argument('--out-dir', default='work_dirs/remote_vg/vis_test')
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preds = load_jsonl(args.pred)
    gt = load_gt(args.gt)

    rendered = []
    for idx, item in enumerate(preds[:args.limit]):
        fname = item['filename']
        image_path = Path(args.img_root) / fname
        out_path = Path(args.out_dir) / f'{idx:03d}_{fname}'
        render_one(image_path, item, gt.get(fname), out_path)
        rendered.append(out_path)

    sheet_path = Path(args.out_dir) / 'contact_sheet.png'
    build_contact_sheet(rendered, sheet_path, cols=2, thumb_width=800)
    print(f'saved {len(rendered)} images to {args.out_dir}')
    print(f'contact sheet: {sheet_path}')


if __name__ == '__main__':
    main()
