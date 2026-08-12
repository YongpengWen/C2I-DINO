# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, OptInstanceList
from ..utils import multi_apply
from .grounding_dino_head import GroundingDINOHead


@MODELS.register_module()
class GroundingDINOSpatialContrastHead(GroundingDINOHead):
    """Grounding DINO head with matching-query spatial contrastive loss.

    The extra loss is only applied to the final decoder matching queries. It
    mines unmatched queries that have low IoU with the current GT target but
    high text score for the same expression, then enforces a margin between
    the matched positive query and those spatially confusing negatives.
    """

    def __init__(self,
                 spatial_contrast_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.spatial_contrast_cfg = spatial_contrast_cfg or dict(enable=False)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function with spatial contrast on final matching queries."""
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'batch_gt_instances_ignore=None.'

        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        if self.spatial_contrast_cfg.get('enable', False):
            loss_dict['loss_spatial_contrast'] = self.loss_spatial_contrast(
                all_layers_matching_cls_scores[-1],
                all_layers_matching_bbox_preds[-1],
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas)
            loss_dict.update(self._spatial_contrast_stats)

        for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in enumerate(
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i

        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores,
                    enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i,
                                loss_iou_i) in enumerate(
                                    zip(dn_losses_cls[:-1],
                                        dn_losses_bbox[:-1],
                                        dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_spatial_contrast(self, cls_scores: Tensor, bbox_preds: Tensor,
                              batch_gt_instances: InstanceList,
                              batch_img_metas: List[dict]) -> Tensor:
        """Rank hard negatives down and suppress nearby background queries."""
        cls_scores_list = [cls_scores[i] for i in range(cls_scores.size(0))]
        bbox_preds_list = [bbox_preds[i] for i in range(bbox_preds.size(0))]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        labels_list, _, bbox_targets_list, bbox_weights_list, _, _ = \
            cls_reg_targets

        cfg = self.spatial_contrast_cfg
        margin = float(cfg.get('margin', 0.2))
        loss_weight = float(cfg.get('loss_weight', 0.1))
        rank_weight = float(cfg.get('rank_weight', 1.0))
        suppress_weight = float(cfg.get('suppress_weight', 0.5))
        neg_iou_thr = float(cfg.get('neg_iou_thr', 0.3))
        topk = int(cfg.get('topk', 10))
        score_type = cfg.get('score_type', 'logit')
        phrase_score_aggregation = cfg.get('phrase_score_aggregation', 'max')
        min_neg_score = cfg.get('min_neg_score', None)
        exclude_gt_iou_thr = cfg.get('exclude_gt_iou_thr', None)

        losses = []
        positive_queries = 0
        candidate_negatives = 0
        selected_negatives = 0
        for img_id, img_meta in enumerate(batch_img_metas):
            img_scores = cls_scores[img_id]
            if score_type == 'sigmoid':
                img_scores = img_scores.sigmoid()
            elif score_type != 'logit':
                raise ValueError('score_type must be "logit" or "sigmoid", '
                                 f'but got {score_type}.')

            img_boxes = self._decode_boxes_to_img_space(
                bbox_preds[img_id], img_meta)
            img_targets = self._decode_boxes_to_img_space(
                bbox_targets_list[img_id], img_meta)
            img_labels = labels_list[img_id]
            img_bbox_weights = bbox_weights_list[img_id]

            pos_inds = torch.nonzero(
                img_bbox_weights.sum(-1) > 0,
                as_tuple=False).squeeze(-1)
            if pos_inds.numel() == 0:
                continue

            neg_mask = img_bbox_weights.sum(-1) == 0
            if exclude_gt_iou_thr is not None and \
                    len(batch_gt_instances[img_id].bboxes) > 0:
                max_gt_ious = bbox_overlaps(
                    img_boxes, batch_gt_instances[img_id].bboxes).max(dim=1)[0]
                neg_mask = neg_mask & (max_gt_ious < float(exclude_gt_iou_thr))

            for pos_ind in pos_inds:
                token_mask = img_labels[pos_ind] > 0
                if not token_mask.any():
                    continue
                positive_queries += 1

                pos_score = self._aggregate_phrase_scores(
                    img_scores[pos_ind, token_mask], phrase_score_aggregation)
                if not torch.isfinite(pos_score):
                    continue

                target_box = img_targets[pos_ind].unsqueeze(0)
                target_ious = bbox_overlaps(img_boxes, target_box).squeeze(-1)
                neg_inds = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
                if neg_inds.numel() == 0:
                    continue

                all_neg_scores = self._aggregate_phrase_scores(
                    img_scores[neg_inds][:, token_mask],
                    phrase_score_aggregation)
                all_neg_ious = target_ious[neg_inds]
                finite_mask = torch.isfinite(all_neg_scores) & torch.isfinite(
                    all_neg_ious)
                all_neg_scores = all_neg_scores[finite_mask]
                all_neg_ious = all_neg_ious[finite_mask]
                if all_neg_scores.numel() == 0:
                    continue

                # SIL should suppress text-confusing queries in the wrong
                # location, not duplicate boxes that already overlap the GT.
                spatial_negative_mask = all_neg_ious < neg_iou_thr
                all_neg_scores = all_neg_scores[spatial_negative_mask]
                all_neg_ious = all_neg_ious[spatial_negative_mask]
                if all_neg_scores.numel() == 0:
                    continue
                candidate_negatives += all_neg_scores.numel()

                hard_keep = torch.zeros_like(all_neg_scores, dtype=torch.bool)
                hard_scores = all_neg_scores
                if min_neg_score is not None:
                    hard_scores = torch.where(
                        hard_scores.detach() > float(min_neg_score),
                        hard_scores,
                        hard_scores.new_full(hard_scores.shape, -float('inf')))
                if topk > 0 and hard_scores.numel() > topk:
                    _, topk_inds = hard_scores.detach().topk(topk)
                    hard_keep[topk_inds] = torch.isfinite(hard_scores[topk_inds])
                else:
                    hard_keep = torch.isfinite(hard_scores)

                keep = hard_keep
                if not keep.any():
                    continue
                selected_negatives += keep.sum().item()

                neg_scores = all_neg_scores[keep]
                neg_weights = torch.ones_like(neg_scores)

                rank_loss = (
                    F.softplus(margin - pos_score + neg_scores) *
                    neg_weights).mean()
                suppress_loss = (
                    F.softplus(neg_scores) * neg_weights).mean()

                losses.append(rank_weight * rank_loss +
                              suppress_weight * suppress_loss)

        stat_denom = max(positive_queries, 1)
        self._spatial_contrast_stats = {
            'spatial_contrast_positive_queries': cls_scores.new_tensor(
                positive_queries / len(batch_img_metas)),
            'spatial_contrast_candidate_negatives': cls_scores.new_tensor(
                candidate_negatives / stat_denom),
            'spatial_contrast_selected_negatives': cls_scores.new_tensor(
                selected_negatives / stat_denom),
        }
        if not losses:
            return cls_scores.new_zeros(())
        return torch.stack(losses).mean() * loss_weight

    @staticmethod
    def _aggregate_phrase_scores(scores: Tensor, aggregation: str) -> Tensor:
        """Aggregate scores from the tokens belonging to one phrase."""
        if aggregation == 'max':
            return scores.max(dim=-1)[0]
        if aggregation == 'mean':
            return scores.mean(dim=-1)
        if aggregation == 'logsumexp':
            return torch.logsumexp(scores, dim=-1) - math.log(scores.size(-1))
        raise ValueError(
            'phrase_score_aggregation must be "max", "mean", or '
            f'"logsumexp", but got {aggregation}.')

    @staticmethod
    def _decode_boxes_to_img_space(bbox_cxcywh: Tensor,
                                   img_meta: dict) -> Tensor:
        """Convert normalized cxcywh boxes to image-space xyxy boxes."""
        img_h, img_w = img_meta['img_shape']
        factor = bbox_cxcywh.new_tensor([img_w, img_h, img_w, img_h])
        return bbox_cxcywh_to_xyxy(bbox_cxcywh) * factor
