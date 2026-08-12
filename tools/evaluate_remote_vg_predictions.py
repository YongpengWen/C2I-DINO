import argparse
import json
from pathlib import Path


def load_jsonl(path):
    items = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            items[item['filename']] = item
    return items


def as_boxes(boxes):
    if not boxes:
        return []
    if isinstance(boxes[0], (int, float)):
        return [boxes]
    return boxes


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def average_precision(records, total_gt):
    if total_gt == 0:
        return 0.0
    records = sorted(records, key=lambda x: x['score'], reverse=True)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for record in records:
        if record['tp']:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / total_gt)

    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return ap


def evaluate(preds, gts, topk=(1, 5), iou_thrs=None, pr_thrs=None):
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
            gt_boxes = as_boxes(gt_region.get('bbox', []))
            total_gt_boxes += len(gt_boxes)

            pred_region = pred_regions[idx] if idx < len(pred_regions) else {}
            pred_boxes = as_boxes(pred_region.get('bbox', []))
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
                    cur_iou, cur_inter, cur_union = iou_details(
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
                    iou(pred_box, gt_box) >= 0.5
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
                        cur_iou = iou(pred_box, gt_box)
                        if cur_iou > best_iou:
                            best_iou = cur_iou
                            best_gt = gt_idx
                    is_tp = best_gt is not None and best_iou >= thr
                    if is_tp:
                        matched.add(best_gt)
                    ap_records[thr].append({'score': score, 'tp': is_tp})

    results = {
        f'acc@{k}_iou0.5': topk_hits[k] / total_regions
        if total_regions else 0.0
        for k in topk
    }
    for thr in pr_thrs:
        results[f'Pr@{thr:.1f}'] = pr_hits[thr] / total_regions \
            if total_regions else 0.0
    results['meanIoU'] = iou_sum / total_regions if total_regions else 0.0
    results['cumIoU'] = inter_sum / union_sum if union_sum > 0 else 0.0
    ap_values = []
    for thr in iou_thrs:
        ap = average_precision(ap_records[thr], total_gt_boxes)
        results[f'ap@{thr:.2f}'] = ap
        ap_values.append(ap)
    results['map@0.50:0.95'] = sum(ap_values) / len(ap_values)
    results['num_images'] = len(gts)
    results['num_regions'] = total_regions
    results['num_gt_boxes'] = total_gt_boxes
    return results


def iou_details(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0, inter, union
    return inter / union, inter, union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='work_dirs/remote_vg/test_predictions.jsonl')
    parser.add_argument('--gt', default='data/remote_vg/annotations/test_vg.json')
    args = parser.parse_args()

    preds = load_jsonl(Path(args.pred))
    gts = load_jsonl(Path(args.gt))
    results = evaluate(preds, gts)

    print(f'images: {results.pop("num_images")}')
    print(f'regions: {results.pop("num_regions")}')
    print(f'gt boxes: {results.pop("num_gt_boxes")}')
    for key, value in results.items():
        print(f'{key}: {value:.4f}')


if __name__ == '__main__':
    main()
