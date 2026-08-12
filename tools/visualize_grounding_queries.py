# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize the 900 Grounding DINO query boxes on a few samples.

This script saves one image for the language-guided top-900 encoder proposals
and one image for each decoder layer's 900 predicted boxes. It also builds a
contact sheet per sample so the query evolution can be inspected quickly.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from PIL import Image, ImageDraw, ImageFont

from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores,
)
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize Grounding DINO intermediate query boxes.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/query_vis',
        help='Directory to save rendered images.')
    parser.add_argument(
        '--num-images',
        type=int,
        default=5,
        help='Number of dataloader samples to visualize.')
    parser.add_argument(
        '--indices-file',
        default=None,
        help='Optional JSON/text file of dataset indices to visualize.')
    parser.add_argument(
        '--indices',
        default=None,
        help='Comma-separated dataset indices to visualize.')
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start dataset index when no explicit indices are given.')
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device used by the runner/model.')
    parser.add_argument(
        '--highlight-topk',
        type=int,
        default=30,
        help='Draw the top-k query boxes with stronger outlines.')
    parser.add_argument(
        '--line-alpha',
        type=int,
        default=46,
        help='Alpha value for the all-query overlay, in [0, 255].')
    parser.add_argument(
        '--max-side',
        type=int,
        default=1200,
        help='Resize output visualization if the longest side is larger.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=None,
        help='Optional score threshold for the faint all-query overlay.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        default=None,
        help='Reserved placeholder for simple key=value overrides.')
    return parser.parse_args()


def load_indices(path: Optional[str]) -> List[int]:
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


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def resize_for_output(
    image: Image.Image,
    boxes: torch.Tensor,
    max_side: int,
) -> Tuple[Image.Image, torch.Tensor, float]:
    scale = 1.0
    longest = max(image.width, image.height)
    if max_side > 0 and longest > max_side:
        scale = max_side / float(longest)
        image = image.resize(
            (int(round(image.width * scale)), int(round(image.height * scale))))
        boxes = boxes * boxes.new_tensor([scale, scale, scale, scale])
    return image, boxes, scale


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def normalized_to_original_xyxy(
    boxes_cxcywh: torch.Tensor,
    metainfo: Dict,
) -> torch.Tensor:
    img_h, img_w = metainfo['img_shape'][:2]
    boxes = cxcywh_to_xyxy(boxes_cxcywh.detach().float().cpu())
    boxes[:, 0::2] *= img_w
    boxes[:, 1::2] *= img_h
    boxes[:, 0::2].clamp_(min=0, max=img_w)
    boxes[:, 1::2].clamp_(min=0, max=img_h)

    scale_factor = metainfo.get('scale_factor', None)
    if scale_factor is not None:
        scale = torch.as_tensor(scale_factor, dtype=torch.float32)
        if scale.numel() == 2:
            boxes = boxes / scale.repeat(2)
        elif scale.numel() == 4:
            boxes = boxes / scale
    return boxes


def gt_boxes_from_sample(data_sample) -> torch.Tensor:
    if not hasattr(data_sample, 'gt_instances'):
        return torch.empty((0, 4), dtype=torch.float32)
    gt_instances = data_sample.gt_instances
    if not hasattr(gt_instances, 'bboxes') or len(gt_instances.bboxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    bboxes = gt_instances.bboxes
    if hasattr(bboxes, 'tensor'):
        bboxes = bboxes.tensor
    return bboxes.detach().float().cpu()


def query_scores(
    cls_score: torch.Tensor,
    token_positive_map: Optional[Dict],
) -> torch.Tensor:
    cls_score = cls_score.detach().sigmoid()
    if token_positive_map is not None:
        cls_score = convert_grounding_to_cls_scores(
            cls_score[None], [token_positive_map])[0]
    scores, _ = cls_score.max(-1)
    return scores.float().cpu()


def bbox_iou(box: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[float, int]:
    if len(gt_boxes) == 0:
        return 0.0, -1
    box = box.float()
    gt_boxes = gt_boxes.float()
    x1 = torch.maximum(box[0], gt_boxes[:, 0])
    y1 = torch.maximum(box[1], gt_boxes[:, 1])
    x2 = torch.minimum(box[2], gt_boxes[:, 2])
    y2 = torch.minimum(box[3], gt_boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (
        gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    union = area_box + area_gt - inter
    ious = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    best_idx = int(torch.argmax(ious).item())
    return float(ious[best_idx].item()), best_idx


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    pad = 5
    bbox = draw.textbbox(xy, text, font=font)
    bg = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(bg, fill=(255, 255, 255))
    draw.text(xy, text, fill=(20, 24, 28), font=font)


def draw_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    color: Tuple[int, int, int, int],
    width: int = 1,
) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def render_stage(
    image_path: str,
    boxes: torch.Tensor,
    scores: Optional[torch.Tensor],
    gt_boxes: torch.Tensor,
    out_path: Path,
    title: str,
    caption: str,
    highlight_topk: int,
    line_alpha: int,
    max_side: int,
    score_thr: Optional[float],
) -> None:
    image = Image.open(image_path).convert('RGB')
    boxes = boxes.clone()
    gt_boxes = gt_boxes.clone()
    image, boxes, scale = resize_for_output(image, boxes, max_side)
    if len(gt_boxes) > 0 and scale != 1.0:
        gt_boxes *= scale

    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    all_color = (38, 104, 210, max(0, min(255, line_alpha)))
    strong_color = (220, 53, 69, 230)
    gt_color = (20, 145, 75, 255)

    keep = torch.ones((len(boxes), ), dtype=torch.bool)
    if scores is not None and score_thr is not None:
        keep = scores >= score_thr
    for box in boxes[keep]:
        draw_rectangle(draw_overlay, box.tolist(), all_color, width=1)

    if scores is None:
        strong_indices = torch.arange(min(highlight_topk, len(boxes)))
    else:
        strong_indices = torch.argsort(scores, descending=True)[:highlight_topk]
    for rank, idx in enumerate(strong_indices.tolist()):
        draw_rectangle(draw_overlay, boxes[idx].tolist(), strong_color, width=2)
        if rank < 8:
            x1, y1, _, _ = boxes[idx].tolist()
            label = f'{idx}'
            if scores is not None:
                label += f' {float(scores[idx]):.2f}'
            draw_label(draw_overlay, (int(x1) + 3, int(y1) + 3), label,
                       get_font(14))

    for box in gt_boxes:
        draw_rectangle(draw_overlay, box.tolist(), gt_color, width=3)

    image = Image.alpha_composite(image.convert('RGBA'), overlay)
    draw = ImageDraw.Draw(image)
    font = get_font(18)
    small_font = get_font(15)
    header = f'{title} | query boxes: {len(boxes)} | strong: top {min(highlight_topk, len(boxes))}'
    draw_label(draw, (8, 8), header, font)
    if caption:
        draw_label(draw, (8, 38), caption[:160], small_font)
    if len(gt_boxes) > 0:
        draw_label(draw, (8, image.height - 28), 'green: GT', small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert('RGB').save(out_path)


def build_contact_sheet(image_paths: Iterable[Path], out_path: Path) -> None:
    images = [Image.open(path).convert('RGB') for path in image_paths]
    if not images:
        return
    thumb_w = 480
    thumbs = []
    for image in images:
        ratio = thumb_w / float(image.width)
        thumbs.append(image.resize((thumb_w, int(round(image.height * ratio)))))
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    row_heights = []
    for row in range(rows):
        row_imgs = thumbs[row * cols:(row + 1) * cols]
        row_heights.append(max(img.height for img in row_imgs))
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def prepare_cfg(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.device = args.device
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.out_dir
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_dataloader.drop_last = False
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


def collect_text_inputs(model, batch_inputs, batch_data_samples):
    text_prompts = []
    enhanced_text_prompts = []
    tokens_positives = []
    for data_sample in batch_data_samples:
        text_prompts.append(data_sample.text)
        enhanced_text_prompts.append(
            data_sample.caption_prompt if 'caption_prompt' in data_sample else None)
        tokens_positives.append(data_sample.get('tokens_positive', None))

    custom_entities = batch_data_samples[0].custom_entities \
        if 'custom_entities' in batch_data_samples[0] else False
    prompts = [
        model.get_tokens_positive_and_prompts(text_prompt, custom_entities,
                                              enhanced_text_prompt,
                                              tokens_positive)
        for text_prompt, enhanced_text_prompt, tokens_positive in zip(
            text_prompts, enhanced_text_prompts, tokens_positives)
    ]
    token_positive_maps, text_prompts, _, entities = zip(*prompts)
    text_dict = model.language_model(list(text_prompts))
    if model.text_feat_map is not None:
        text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])
    for data_sample, token_positive_map in zip(batch_data_samples,
                                               token_positive_maps):
        data_sample.token_positive_map = token_positive_map
    return text_dict, list(text_prompts), list(entities)


@torch.no_grad()
def extract_queries(model, data: Dict) -> Tuple[Dict, List]:
    data = model.data_preprocessor(data, False)
    batch_inputs = data['inputs']
    batch_data_samples = data['data_samples']
    text_dict, text_prompts, entities = collect_text_inputs(
        model, batch_inputs, batch_data_samples)
    visual_feats = model.extract_feat(batch_inputs)
    head_inputs = model.forward_transformer(
        visual_feats, text_dict, batch_data_samples)
    layer_cls, layer_boxes = model.bbox_head(
        head_inputs['hidden_states'],
        head_inputs['references'],
        head_inputs['memory_text'],
        head_inputs['text_token_mask'],
    )
    return dict(
        references=head_inputs['references'],
        layer_cls=layer_cls,
        layer_boxes=layer_boxes,
        text_prompts=text_prompts,
        entities=entities,
    ), batch_data_samples


def main() -> None:
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = prepare_cfg(args)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)

    model, dataset = build_model_and_dataset(cfg, args.checkpoint, args.device)
    model = unwrap_model(model)

    metadata = []
    rendered_count = 0
    if args.indices_file:
        target_indices = load_indices(args.indices_file)
    elif args.indices:
        target_indices = [int(x) for x in args.indices.split(',') if x.strip()]
    else:
        end = min(len(dataset), args.start_index + args.num_images)
        target_indices = list(range(args.start_index, end))
    target_indices = target_indices[:args.num_images]

    for dataset_index in target_indices:
        if rendered_count >= args.num_images:
            break
        item = dataset[dataset_index]
        data = dict(
            inputs=[item['inputs']],
            data_samples=[item['data_samples']],
        )
        query_data, batch_data_samples = extract_queries(model, data)

        data_sample = batch_data_samples[0]
        metainfo = data_sample.metainfo
        image_path = metainfo['img_path']
        stem = Path(image_path).stem
        sample_dir = out_dir / f'{rendered_count:02d}_{stem}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        token_positive_map = data_sample.get('token_positive_map', None)
        caption = query_data['text_prompts'][0]
        gt_boxes = gt_boxes_from_sample(data_sample)

        stage_paths = []
        init_boxes = normalized_to_original_xyxy(query_data['references'][0][0],
                                                 metainfo)
        init_path = sample_dir / '00_encoder_top900.png'
        render_stage(
            image_path=image_path,
            boxes=init_boxes,
            scores=None,
            gt_boxes=gt_boxes,
            out_path=init_path,
            title='encoder language-guided top-900',
            caption=caption,
            highlight_topk=args.highlight_topk,
            line_alpha=args.line_alpha,
            max_side=args.max_side,
            score_thr=None,
        )
        stage_paths.append(init_path)

        top1_info = None
        for layer_id in range(query_data['layer_boxes'].shape[0]):
            boxes = normalized_to_original_xyxy(
                query_data['layer_boxes'][layer_id, 0], metainfo)
            scores = query_scores(query_data['layer_cls'][layer_id, 0],
                                  token_positive_map)
            title = f'decoder layer {layer_id + 1} top-900'
            top1_info = None
            if layer_id == query_data['layer_boxes'].shape[0] - 1 and len(scores) > 0:
                top1_idx = int(torch.argmax(scores).item())
                top1_iou, best_gt_idx = bbox_iou(boxes[top1_idx], gt_boxes)
                top1_info = dict(
                    query_index=top1_idx,
                    score=float(scores[top1_idx].item()),
                    bbox=boxes[top1_idx].tolist(),
                    iou=top1_iou,
                    best_gt_idx=best_gt_idx,
                )
                title = (
                    f'decoder layer {layer_id + 1} final top-900 '
                    f'| top1 q={top1_idx} score={top1_info["score"]:.3f} '
                    f'IoU={top1_iou:.3f}')
            stage_path = sample_dir / f'{layer_id + 1:02d}_decoder_layer_{layer_id + 1}.png'
            render_stage(
                image_path=image_path,
                boxes=boxes,
                scores=scores,
                gt_boxes=gt_boxes,
                out_path=stage_path,
                title=title,
                caption=caption,
                highlight_topk=args.highlight_topk,
                line_alpha=args.line_alpha,
                max_side=args.max_side,
                score_thr=args.score_thr,
            )
            stage_paths.append(stage_path)

        sheet_path = sample_dir / 'contact_sheet.png'
        build_contact_sheet(stage_paths, sheet_path)
        metadata.append(
            dict(
                index=rendered_count,
                dataset_index=dataset_index,
                image_path=image_path,
                caption=caption,
                ori_shape=metainfo.get('ori_shape'),
                img_shape=metainfo.get('img_shape'),
                scale_factor=metainfo.get('scale_factor'),
                gt_boxes=gt_boxes.tolist(),
                final_top1=top1_info,
                query_count=int(init_boxes.shape[0]),
                stage_images=[str(path) for path in stage_paths],
                contact_sheet=str(sheet_path),
            ))
        print(f'[{rendered_count + 1}/{args.num_images}] saved {sheet_path}')
        rendered_count += 1

    metadata_path = out_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'saved metadata: {metadata_path}')


if __name__ == '__main__':
    main()
