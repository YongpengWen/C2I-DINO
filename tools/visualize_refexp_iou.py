# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize referring-expression predictions with GT and IoU.

The metric in this repo evaluates the highest-score prediction against the GT
box(es). This script mirrors that behavior, ranks samples by top-1 IoU, and
saves low-IoU visualizations for diagnosis.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from PIL import Image, ImageDraw, ImageFont

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize top referring-expression predictions and IoU.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--ann-file',
        default=None,
        help='COCO/MDETR-style annotation file. Defaults to test_evaluator.ann_file.')
    parser.add_argument(
        '--dataset-ann-file',
        default=None,
        help='Override the test dataset annotation path in the config.')
    parser.add_argument(
        '--dataset-data-root',
        default=None,
        help='Override the test dataset data_root in the config.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/refexp_iou_vis',
        help='Output directory.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=40,
        help='Number of dataset samples to run. Use -1 for all.')
    parser.add_argument(
        '--random-samples',
        type=int,
        default=0,
        help='Randomly sample this many dataset indices before inference.')
    parser.add_argument(
        '--sample-indices-file',
        default=None,
        help='Optional JSON/text file of dataset indices to evaluate.')
    parser.add_argument(
        '--seed',
        type=int,
        default=2026,
        help='Random seed used with --random-samples.')
    parser.add_argument(
        '--num-low',
        type=int,
        default=20,
        help='Number of lowest-IoU samples to render.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Inference batch size.')
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Inference device.')
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='Number of high-score predictions to draw.')
    parser.add_argument(
        '--max-side',
        type=int,
        default=1400,
        help='Resize saved visualization if the longest side is larger.')
    return parser.parse_args()


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_gt(ann_file: str) -> Dict[int, Dict]:
    data = json.load(open(ann_file, 'r'))
    images = {img['id']: img for img in data['images']}
    anns_by_image: Dict[int, List] = {}
    for ann in data['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    out = {}
    for img_id, img in images.items():
        boxes = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann['bbox']
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
        out[img_id] = dict(image=img, gt_boxes=boxes)
    return out


def bbox_iou(box: Sequence[float], gt_boxes: Sequence[Sequence[float]]) -> Tuple[float, int]:
    if not gt_boxes:
        return 0.0, -1
    box_arr = np.asarray(box, dtype=np.float32)
    gt_arr = np.asarray(gt_boxes, dtype=np.float32)
    x1 = np.maximum(box_arr[0], gt_arr[:, 0])
    y1 = np.maximum(box_arr[1], gt_arr[:, 1])
    x2 = np.minimum(box_arr[2], gt_arr[:, 2])
    y2 = np.minimum(box_arr[3], gt_arr[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_pred = max(0.0, float(box_arr[2] - box_arr[0])) * max(
        0.0, float(box_arr[3] - box_arr[1]))
    area_gt = np.maximum(0.0, gt_arr[:, 2] - gt_arr[:, 0]) * np.maximum(
        0.0, gt_arr[:, 3] - gt_arr[:, 1])
    union = area_pred + area_gt - inter
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    best = int(np.argmax(ious))
    return float(ious[best]), best


def prepare_cfg(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.device = args.device
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_dataloader.drop_last = False
    if args.dataset_ann_file:
        cfg.test_dataloader.dataset.ann_file = args.dataset_ann_file
    if args.dataset_data_root:
        cfg.test_dataloader.dataset.data_root = args.dataset_data_root
    return cfg


def build_model_and_dataset(cfg: Config, checkpoint: str, device: str):
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model)
    checkpoint_data = load_checkpoint(
        model, checkpoint, map_location=device, revise_keys=[(r'^module\.', '')])
    model.dataset_meta = checkpoint_data.get('meta', {}).get('dataset_meta', None)
    model.to(device)
    model.eval()
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    return model, dataset


def load_indices(path: str) -> List[int]:
    if path is None:
        return []
    text = Path(path).read_text().strip()
    if not text:
        return []
    if text[0] in '[{':
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get('indices', data.get('samples', []))
        indices = []
        for item in data:
            if isinstance(item, dict):
                indices.append(int(item.get('dataset_index', item.get('index'))))
            else:
                indices.append(int(item))
        return indices
    return [int(line.strip()) for line in text.splitlines() if line.strip()]


def make_batch(items: Sequence[Dict]) -> Dict:
    return dict(
        inputs=[item['inputs'] for item in items],
        data_samples=[item['data_samples'] for item in items],
    )


@torch.no_grad()
def predict_batch(model, items: Sequence[Dict]):
    data = model.data_preprocessor(make_batch(items), False)
    return model.predict(data['inputs'], data['data_samples'], rescale=True)


def draw_box(draw: ImageDraw.ImageDraw,
             box: Sequence[float],
             color: Tuple[int, int, int],
             width: int) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def draw_label(draw: ImageDraw.ImageDraw,
               xy: Tuple[int, int],
               text: str,
               color: Tuple[int, int, int],
               font: ImageFont.ImageFont) -> None:
    x, y = xy
    pad = 4
    bbox = draw.textbbox((x, y), text, font=font)
    bg = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(bg, fill=(255, 255, 255))
    draw.rectangle(bg, outline=color, width=2)
    draw.text((x, y), text, fill=color, font=font)


def resize_boxes(image: Image.Image,
                 boxes: List[List[float]],
                 max_side: int) -> Tuple[Image.Image, List[List[float]], float]:
    longest = max(image.width, image.height)
    if max_side <= 0 or longest <= max_side:
        return image, boxes, 1.0
    scale = max_side / float(longest)
    resized = image.resize(
        (int(round(image.width * scale)), int(round(image.height * scale))))
    scaled_boxes = [[v * scale for v in box] for box in boxes]
    return resized, scaled_boxes, scale


def render_one(result: Dict, out_path: Path, topk: int, max_side: int) -> None:
    image = Image.open(result['image_path']).convert('RGB')
    boxes = result['gt_boxes'] + [pred['bbox'] for pred in result['predictions'][:topk]]
    image, scaled, _ = resize_boxes(image, boxes, max_side)
    gt_boxes = scaled[:len(result['gt_boxes'])]
    pred_boxes = scaled[len(result['gt_boxes']):]

    draw = ImageDraw.Draw(image)
    font = get_font(18)
    small = get_font(15)
    green = (24, 150, 70)
    red = (220, 53, 69)
    orange = (245, 135, 31)

    for idx, box in enumerate(gt_boxes):
        draw_box(draw, box, green, 4)
        if idx == result['best_gt_idx']:
            draw_label(draw, (int(box[0]) + 4, int(box[1]) + 4), 'GT best',
                       green, small)
        elif idx == 0:
            draw_label(draw, (int(box[0]) + 4, int(box[1]) + 4), 'GT', green,
                       small)

    for idx, box in enumerate(pred_boxes):
        pred = result['predictions'][idx]
        if idx == 0:
            draw_box(draw, box, red, 4)
            label = f"top1 score={pred['score']:.3f} IoU={result['top1_iou']:.3f}"
            draw_label(draw, (int(box[0]) + 4, int(box[1]) + 4), label, red,
                       small)
        else:
            draw_box(draw, box, orange, 2)
            draw_label(draw, (int(box[0]) + 3, int(box[1]) + 3),
                       f"{idx + 1}:{pred['score']:.2f}", orange, small)

    header = f"img_id={result['img_id']} | top1 IoU={result['top1_iou']:.3f}"
    draw_label(draw, (8, 8), header, red if result['top1_iou'] < 0.5 else green,
               font)
    draw_label(draw, (8, 40), result['caption'][:180], (20, 24, 28), small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def build_contact_sheet(image_paths: Iterable[Path], out_path: Path) -> None:
    images = [Image.open(path).convert('RGB') for path in image_paths]
    if not images:
        return
    thumb_w = 520
    thumbs = []
    for image in images:
        ratio = thumb_w / float(image.width)
        thumbs.append(image.resize((thumb_w, int(round(image.height * ratio)))))
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    row_heights = [
        max(img.height for img in thumbs[row * cols:(row + 1) * cols])
        for row in range(rows)
    ]
    sheet = Image.new('RGB', (thumb_w * cols, sum(row_heights)), (245, 245, 245))
    y = 0
    idx = 0
    for row_h in row_heights:
        x = 0
        for _ in range(cols):
            if idx >= len(thumbs):
                break
            sheet.paste(thumbs[idx], (x, y))
            x += thumb_w
            idx += 1
        y += row_h
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = prepare_cfg(args)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    ann_file = args.ann_file or cfg.test_evaluator.ann_file
    gt_by_img = load_gt(ann_file)
    model, dataset = build_model_and_dataset(cfg, args.checkpoint, args.device)

    if args.sample_indices_file:
        sample_indices = load_indices(args.sample_indices_file)
    elif args.random_samples > 0:
        rng = random.Random(args.seed)
        sample_indices = rng.sample(
            range(len(dataset)), min(args.random_samples, len(dataset)))
    else:
        total = len(dataset) if args.max_samples < 0 else min(
            args.max_samples, len(dataset))
        sample_indices = list(range(total))
    if args.max_samples >= 0:
        sample_indices = sample_indices[:args.max_samples]
    total = len(sample_indices)
    results = []
    for start in range(0, total, args.batch_size):
        end = min(total, start + args.batch_size)
        batch_indices = sample_indices[start:end]
        items = [dataset[idx] for idx in batch_indices]
        samples = predict_batch(model, items)
        for dataset_index, data_sample in zip(batch_indices, samples):
            img_id = int(data_sample.img_id)
            gt_info = gt_by_img.get(img_id, dict(gt_boxes=[]))
            pred = data_sample.pred_instances
            bboxes = pred.bboxes.detach().cpu().numpy()
            scores = pred.scores.detach().cpu().numpy()
            order = np.argsort(-scores)
            predictions = []
            for idx in order[:max(args.topk, 1)]:
                predictions.append(
                    dict(bbox=bboxes[idx].astype(float).tolist(),
                         score=float(scores[idx])))
            if predictions:
                top1_iou, best_gt_idx = bbox_iou(predictions[0]['bbox'],
                                                 gt_info['gt_boxes'])
            else:
                top1_iou, best_gt_idx = 0.0, -1
            results.append(
                dict(
                    dataset_index=int(dataset_index),
                    img_id=img_id,
                    image_path=data_sample.img_path,
                    caption=data_sample.text,
                    gt_boxes=gt_info['gt_boxes'],
                    best_gt_idx=best_gt_idx,
                    top1_iou=top1_iou,
                    predictions=predictions,
                ))
        print(f'processed {end}/{total}', flush=True)

    results_sorted = sorted(results, key=lambda x: x['top1_iou'])
    low_results = results_sorted[:args.num_low]
    rendered = []
    low_dir = out_dir / 'low_iou'
    for rank, result in enumerate(low_results):
        stem = Path(result['image_path']).stem
        out_path = low_dir / f'{rank:03d}_iou_{result["top1_iou"]:.3f}_{stem}.png'
        render_one(result, out_path, args.topk, args.max_side)
        rendered.append(out_path)

    with open(out_dir / 'pred_iou_results.json', 'w') as f:
        json.dump(results_sorted, f, indent=2)
    build_contact_sheet(rendered, out_dir / 'low_iou_contact_sheet.png')

    ious = [r['top1_iou'] for r in results]
    summary = dict(
        ann_file=ann_file,
        checkpoint=args.checkpoint,
        samples=total,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        pr_05=float(np.mean([iou >= 0.5 for iou in ious])) if ious else 0.0,
        pr_07=float(np.mean([iou >= 0.7 for iou in ious])) if ious else 0.0,
        low_iou_count=len(low_results),
    )
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'saved visualizations to {low_dir}', flush=True)
    print(f'contact sheet: {out_dir / "low_iou_contact_sheet.png"}', flush=True)


if __name__ == '__main__':
    main()
