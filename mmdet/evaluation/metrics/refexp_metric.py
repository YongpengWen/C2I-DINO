# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import get_local_path
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from ..functional import bbox_overlaps


@METRICS.register_module()
class RefExpMetric(BaseMetric):
    default_prefix: Optional[str] = 'refexp'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: str = 'bbox',
                 topk=(1, 5, 10),
                 iou_thrs: float = 0.5,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.topk = topk
        self.iou_thrs = iou_thrs

        with get_local_path(ann_file) as local_path:
            self.coco = COCO(local_path)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        dataset2score = {
            'refcoco': {k: 0.0
                        for k in self.topk},
            'refcoco+': {k: 0.0
                         for k in self.topk},
            'refcocog': {k: 0.0
                         for k in self.topk},
        }
        dataset2count = {'refcoco': 0.0, 'refcoco+': 0.0, 'refcocog': 0.0}

        for result in results:
            img_id = result['img_id']

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            assert len(ann_ids) == 1
            img_info = self.coco.loadImgs(img_id)[0]
            target = self.coco.loadAnns(ann_ids[0])

            target_bbox = target[0]['bbox']
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            iou = bbox_overlaps(result['bboxes'],
                                np.array(converted_bbox).reshape(-1, 4))
            for k in self.topk:
                if max(iou[:k]) >= self.iou_thrs:
                    dataset2score[img_info['dataset_name']][k] += 1.0
            dataset2count[img_info['dataset_name']] += 1.0

        for key, value in dataset2score.items():
            for k in self.topk:
                try:
                    value[k] /= dataset2count[key]
                except Exception as e:
                    print(e)

        results = {}
        mean_precision = 0.0
        for key, value in dataset2score.items():
            results[key] = sorted([v for k, v in value.items()])
            mean_precision += sum(results[key])
            logger.info(
                f' Dataset: {key} - Precision @ 1, 5, 10: {results[key]}')

        # `mean_precision` key is used for saving the best checkpoint
        out_results = {'mean_precision': mean_precision / 9.0}

        for i, k in enumerate(self.topk):
            out_results[f'refcoco_precision@{k}'] = results['refcoco'][i]
        for i, k in enumerate(self.topk):
            out_results[f'refcoco+_precision@{k}'] = results['refcoco+'][i]
        for i, k in enumerate(self.topk):
            out_results[f'refcocog_precision@{k}'] = results['refcocog'][i]
        return out_results


@METRICS.register_module()
class MultiBoxRefExpMetric(BaseMetric):
    """OPT-RSVG style referring-expression metric.

    The official OPT-RSVG code evaluates one predicted box per expression and
    reports Pr@0.5..0.9, meanIoU and cumulative IoU. For expressions with
    multiple annotated boxes, this implementation uses the GT box with the best
    IoU to the top-scoring prediction.
    """
    default_prefix: Optional[str] = 'refexp'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: str = 'bbox',
                 iou_thrs=(0.5, 0.6, 0.7, 0.8, 0.9),
                 dataset_name: str = 'opt_rsvg',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.iou_thrs = iou_thrs
        self.dataset_name = dataset_name

        with get_local_path(ann_file) as local_path:
            self.coco = COCO(local_path)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            self.results.append(dict(
                img_id=data_sample['img_id'],
                bboxes=pred['bboxes'].cpu().numpy(),
                scores=pred['scores'].cpu().numpy()))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        hits = {thr: 0.0 for thr in self.iou_thrs}
        iou_sum = 0.0
        inter_sum = 0.0
        union_sum = 0.0
        count = 0.0

        for result in results:
            ann_ids = self.coco.getAnnIds(imgIds=result['img_id'])
            if not ann_ids:
                continue
            anns = self.coco.loadAnns(ann_ids)
            gt_bboxes = []
            for ann in anns:
                x1, y1, w, h = ann['bbox']
                gt_bboxes.append([x1, y1, x1 + w, y1 + h])
            if not gt_bboxes:
                continue

            if len(result['bboxes']) == 0:
                best_iou = 0.0
                best_inter = 0.0
                best_union = 0.0
            else:
                top_idx = int(np.argmax(result['scores']))
                pred_bbox = result['bboxes'][top_idx]
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                ious = bbox_overlaps(
                    pred_bbox.reshape(1, 4), gt_bboxes).reshape(-1)
                best_gt_idx = int(np.argmax(ious))
                best_iou = float(ious[best_gt_idx])
                best_inter, best_union = self._inter_union(
                    pred_bbox, gt_bboxes[best_gt_idx])

            for thr in self.iou_thrs:
                if best_iou >= thr:
                    hits[thr] += 1.0
            iou_sum += best_iou
            inter_sum += best_inter
            union_sum += best_union
            count += 1.0

        out_results = {}
        for thr in self.iou_thrs:
            out_results[f'{self.dataset_name}_Pr@{thr:g}'] = (
                hits[thr] / count if count else 0.0)
        out_results[f'{self.dataset_name}_meanIoU'] = (
            iou_sum / count if count else 0.0)
        out_results[f'{self.dataset_name}_cumIoU'] = (
            inter_sum / union_sum if union_sum > 0 else 0.0)
        out_results['mean_precision'] = out_results[
            f'{self.dataset_name}_Pr@0.5']

        logger.info(
            f'Dataset: {self.dataset_name} - '
            f'Pr: {[out_results[f"{self.dataset_name}_Pr@{thr:g}"] for thr in self.iou_thrs]}, '
            f'meanIoU: {out_results[f"{self.dataset_name}_meanIoU"]}, '
            f'cumIoU: {out_results[f"{self.dataset_name}_cumIoU"]}')
        return out_results

    @staticmethod
    def _inter_union(box1, box2):
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
            0.0, float(box1[3]) - float(box1[1]))
        area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
            0.0, float(box2[3]) - float(box2[1]))
        union = area1 + area2 - inter
        return inter, union
