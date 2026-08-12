# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from typing import Any, Optional, Sequence

from mmcv.ops import batched_nms
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log

from mmdet.registry import METRICS

try:
    import jsonlines
except ImportError:
    jsonlines = None


@METRICS.register_module()
class DumpODVGResults(BaseMetric):
    default_prefix: Optional[str] = 'pl_odvg'

    def __init__(self,
                 outfile_path,
                 img_prefix: str,
                 ann_file: Optional[str] = None,
                 score_thr: float = 0.1,
                 collect_device: str = 'cpu',
                 nms_thr: float = 0.5,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.outfile_path = outfile_path
        self.ann_file = ann_file
        self.score_thr = score_thr
        self.img_prefix = img_prefix
        self.nms_thr = nms_thr

        if jsonlines is None:
            raise ImportError('Please run "pip install jsonlines" to install '
                              'this package.')

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = {}

            filename = data_sample['img_path']
            filename = filename.replace(self.img_prefix, '')
            if filename.startswith('/'):
                filename = filename[1:]
            result['filename'] = filename

            height = data_sample['ori_shape'][0]
            width = data_sample['ori_shape'][1]
            result['height'] = height
            result['width'] = width

            pred_instances = data_sample['pred_instances']

            bboxes = pred_instances['bboxes'].cpu()
            scores = pred_instances['scores'].cpu()
            labels = pred_instances['labels'].cpu()

            bboxes = bboxes[scores > self.score_thr]
            labels = labels[scores > self.score_thr]
            scores = scores[scores > self.score_thr]

            gt_instances = data_sample.get('gt_instances')
            if 'tokens_positive' in data_sample or (
                    gt_instances is not None and 'labels' in gt_instances):
                task = 'vg'
            else:
                task = 'od'

            if task == 'od':
                classes_name = data_sample['text']
                result['detection'] = {}

                if len(bboxes) > 0:
                    det_bboxes, keep = batched_nms(
                        bboxes, scores, labels,
                        dict(type='nms', iou_threshold=self.nms_thr))
                    _scores = det_bboxes[:, -1]
                    _bboxes = det_bboxes[:, :-1]
                    _labels = labels[keep]

                    instances = []
                    _bboxes = _bboxes.numpy().tolist()
                    _scores = _scores.numpy().tolist()
                    _labels = _labels.numpy().tolist()
                    for bbox, score, label in zip(_bboxes, _scores, _labels):
                        round_bbox = [round(b, 2) for b in bbox]
                        round_score = round(score, 2)
                        instances.append({
                            'bbox': round_bbox,
                            'score': round_score,
                            'label': label,
                            'category': classes_name[label]
                        })
                    result['detection']['instances'] = instances
                else:
                    result['detection']['instances'] = []
                self.results.append(result)
            else:
                caption = data_sample['text']
                result['grounding'] = {}
                result['grounding']['caption'] = caption

                tokens_positive = data_sample.get('tokens_positive')
                if tokens_positive is None:
                    num_regions = int(gt_instances['labels'].max().item() +
                                      1) if len(gt_instances['labels']) else 0
                    tokens_positive = [[] for _ in range(num_regions)]

                region_list = []
                for label, positive in enumerate(tokens_positive):
                    phrase = [caption[pos[0]:pos[1]] for pos in positive]
                    if not phrase:
                        phrase = [caption]

                    _bboxes = bboxes[labels == label]
                    _scores = scores[labels == label]
                    det_bboxes, _ = batched_nms(
                        _bboxes,
                        _scores,
                        None,
                        dict(type='nms', iou_threshold=self.nms_thr),
                        class_agnostic=True)
                    _scores = det_bboxes[:, -1].numpy().tolist()
                    _bboxes = det_bboxes[:, :-1].numpy().tolist()

                    round_bboxes = []
                    for bbox in _bboxes:
                        round_bboxes.append([round(b, 2) for b in bbox])
                    _scores = [[round(s, 2) for s in _scores]]
                    region = {
                        'phrase': phrase,
                        'bbox': round_bboxes,
                        'score': _scores,
                        'tokens_positive': positive
                    }
                    region_list.append(region)
                result['grounding']['regions'] = region_list
                self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        out_dir = os.path.dirname(self.outfile_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with jsonlines.open(self.outfile_path, mode='w') as writer:
            writer.write_all(results)
        print_log(
            f'Results has been saved to {self.outfile_path}.',
            logger='current')
        if self.ann_file is None:
            return {}
        gts = self._load_jsonl(self.ann_file)
        preds = {item['filename']: item for item in results}
        return self._evaluate_vg(preds, gts)

    @staticmethod
    def _load_jsonl(path):
        items = {}
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                items[item['filename']] = item
        return items

    @staticmethod
    def _as_boxes(boxes):
        if not boxes:
            return []
        if isinstance(boxes[0], (int, float)):
            return [boxes]
        return boxes

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area1 = max(0.0, box1[2] - box1[0]) * max(
            0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(
            0.0, box2[3] - box2[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _average_precision(records, total_gt):
        if total_gt == 0:
            return 0.0
        records = sorted(records, key=lambda x: x['score'], reverse=True)
        tp = 0
        fp = 0
        ap = 0.0
        prev_recall = 0.0
        for record in records:
            if record['tp']:
                tp += 1
            else:
                fp += 1
            precision = tp / max(tp + fp, 1)
            recall = tp / total_gt
            ap += precision * max(0.0, recall - prev_recall)
            prev_recall = recall
        return ap

    @staticmethod
    def _iou_details(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area1 = max(0.0, box1[2] - box1[0]) * max(
            0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(
            0.0, box2[3] - box2[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0, inter, union
        return inter / union, inter, union

    def _evaluate_vg(self, preds, gts, topk=(1, 5), iou_thrs=None,
                     pr_thrs=None):
        if iou_thrs is None:
            iou_thrs = [x / 100 for x in range(50, 100, 5)]
        if pr_thrs is None:
            pr_thrs = [0.5, 0.6, 0.7, 0.8, 0.9]

        total_regions = 0
        topk_hits = {k: 0 for k in topk}
        pr_hits = {thr: 0 for thr in pr_thrs}
        iou_sum = 0.0
        inter_sum = 0.0
        union_sum = 0.0
        ap_records = {thr: [] for thr in iou_thrs}
        total_gt_boxes = 0

        for filename, gt_item in gts.items():
            pred_item = preds.get(filename, {})
            gt_regions = gt_item.get('grounding', {}).get('regions', [])
            pred_regions = pred_item.get('grounding', {}).get('regions', [])

            for idx, gt_region in enumerate(gt_regions):
                total_regions += 1
                gt_boxes = self._as_boxes(gt_region.get('bbox', []))
                total_gt_boxes += len(gt_boxes)

                pred_region = (
                    pred_regions[idx] if idx < len(pred_regions) else {})
                pred_boxes = self._as_boxes(pred_region.get('bbox', []))
                scores = pred_region.get('score', [[]])
                scores = scores[0] if scores else []
                scored_boxes = []
                for box_idx, box in enumerate(pred_boxes):
                    score = scores[box_idx] if box_idx < len(scores) else 0.0
                    scored_boxes.append((float(score), box))
                scored_boxes.sort(key=lambda x: x[0], reverse=True)

                top1_iou = 0.0
                top1_inter = 0.0
                top1_union = 0.0
                if scored_boxes and gt_boxes:
                    _, top1_box = scored_boxes[0]
                    for gt_box in gt_boxes:
                        cur_iou, cur_inter, cur_union = self._iou_details(
                            top1_box, gt_box)
                        if cur_iou > top1_iou:
                            top1_iou = cur_iou
                            top1_inter = cur_inter
                            top1_union = cur_union
                iou_sum += top1_iou
                inter_sum += top1_inter
                union_sum += top1_union
                for thr in pr_thrs:
                    if top1_iou >= thr:
                        pr_hits[thr] += 1

                for k in topk:
                    candidates = scored_boxes[:k]
                    hit = any(
                        self._iou(pred_box, gt_box) >= 0.5
                        for _, pred_box in candidates
                        for gt_box in gt_boxes)
                    if hit:
                        topk_hits[k] += 1

                for thr in iou_thrs:
                    matched = set()
                    for score, pred_box in scored_boxes:
                        best_iou = 0.0
                        best_gt = None
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            if gt_idx in matched:
                                continue
                            cur_iou = self._iou(pred_box, gt_box)
                            if cur_iou > best_iou:
                                best_iou = cur_iou
                                best_gt = gt_idx
                        is_tp = best_gt is not None and best_iou >= thr
                        if is_tp:
                            matched.add(best_gt)
                        ap_records[thr].append({
                            'score': score,
                            'tp': is_tp
                        })

        metrics = {
            f'acc@{k}_iou0.5': topk_hits[k] / total_regions
            if total_regions else 0.0
            for k in topk
        }
        for thr in pr_thrs:
            metrics[f'Pr@{thr:.1f}'] = pr_hits[thr] / total_regions \
                if total_regions else 0.0
        metrics['meanIoU'] = iou_sum / total_regions if total_regions else 0.0
        metrics['cumIoU'] = inter_sum / union_sum if union_sum > 0 else 0.0
        ap_values = [
            self._average_precision(ap_records[thr], total_gt_boxes)
            for thr in iou_thrs
        ]
        metrics['map@0.50:0.95'] = sum(ap_values) / len(ap_values)
        return metrics
