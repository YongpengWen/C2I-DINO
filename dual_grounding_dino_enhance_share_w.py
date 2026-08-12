import copy
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.structures import OptSampleList, SampleList
from mmrotate.registry import MODELS
from .grounding_dino import RotatedGroundingDINO
from .grounding_dino_layers import GroundingDinoTransformerEncoder
from .asymmetric_adaptive_cross_model_guidance import WFACModule


@MODELS.register_module()
class SharedDualRotatedGroundingDINOEnhanceAdd(RotatedGroundingDINO): 
    """Two-Dual-branch Rotated Grounding DINO for IR + VI fusion with OBB.

    The main backbone/neck (inherited from RotatedGroundingDINO) processes IR
    images (the primary modality). A second backbone/neck processes VI images
    (the auxiliary modality). Features from both branches are fused via
    element-wise addition after SHARED cross-modal feature enhancement.

    Args:
        backbone_vi (dict): Config for the VI (auxiliary) backbone.
        neck_vi (dict): Config for the VI (auxiliary) neck.
    """

    def __init__(self, backbone_vi, neck_vi, wfac_cfg=None, num_classes=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_vi = MODELS.build(backbone_vi)
        self.neck_vi = MODELS.build(neck_vi)
        
        # 【修改点 1】改用 nn.ModuleList 为 FPN 的 4 个 Level (0~3) 独立分配小波融合参数
        if wfac_cfg is not None:
            self.wfac_layers = nn.ModuleList([
                WFACModule(**wfac_cfg) for _ in range(4)
            ])
        else:
            self.wfac_layers = None
        
        # 类别特定的可学习后缀向量：每个类别8个token，维度256
        self.num_classes = num_classes
        # 使用较小的初始化值，避免数值不稳定
        self.suffix_tokens = nn.Parameter(torch.randn(num_classes, 8, 256) * 0.01)

    def _append_suffix_tokens(self, text_dict, batch_data_samples):
        """根据样本类别拼接对应的可学习后缀向量"""
        device = text_dict['embedded'].device
        batch_size = len(batch_data_samples)
        
        # 获取每个样本的类别 ID
        gt_labels_list = []
        for data_samples in batch_data_samples:
            if hasattr(data_samples, 'gt_instances') and hasattr(data_samples.gt_instances, 'labels'):
                gt_labels = data_samples.gt_instances.labels
                if len(gt_labels) > 0:
                    gt_labels_list.append(gt_labels)
                else:
                    gt_labels_list.append(None)
            else:
                gt_labels_list.append(None)
        
        # 1. 处理嵌入向量
        new_embeddings = []
        for gt_labels in gt_labels_list:
            if gt_labels is not None and len(gt_labels) > 0:
                cls_id = gt_labels[0].item() if hasattr(gt_labels[0], 'item') else int(gt_labels[0])
                cls_id = min(cls_id, self.num_classes - 1)
            else:
                cls_id = 0
            suffix = self.suffix_tokens[cls_id].unsqueeze(0)
            new_embeddings.append(suffix)
        
        embeddings = text_dict['embedded']
        new_embeddings = torch.cat(new_embeddings, dim=0)
        text_dict['embedded'] = torch.cat([embeddings, new_embeddings], dim=1)
        
        # 2. 处理 text_token_mask：拼接 True（有效token）
        masks = text_dict['text_token_mask']
        new_masks = torch.ones(batch_size, 8, dtype=torch.bool, device=device)
        text_dict['text_token_mask'] = torch.cat([masks, new_masks], dim=1)
        
        # 3. 处理 position_ids：重新生成连续的位置索引
        if 'position_ids' in text_dict and text_dict['position_ids'] is not None:
            seq_len = text_dict['embedded'].shape[1]  # 新的长度 L+8
            # 重新生成连续的位置索引 [0, 1, 2, ..., L+7]
            text_dict['position_ids'] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 4. 处理 masks (text_self_attention_masks)
        # masks 的含义：1=可以attend，0=mask out
        if 'masks' in text_dict and text_dict['masks'] is not None:
            orig_masks = text_dict['masks']
            seq_len_m = embeddings.shape[1]
            if orig_masks.dim() == 2:
                # 2D mask: 拼接全1（后缀可以attend）
                new_mask_row = torch.ones(batch_size, 8, dtype=orig_masks.dtype, device=device)
                new_mask_col = torch.ones(batch_size, seq_len_m, dtype=orig_masks.dtype, device=device)
                text_dict['masks'] = torch.cat([
                    torch.cat([orig_masks, new_mask_row], dim=1),
                    torch.cat([new_mask_col, torch.ones(batch_size, 8, dtype=orig_masks.dtype, device=device)], dim=1)
                ], dim=1)
            elif orig_masks.dim() == 3:
                # 3D mask: 拼接全1（后缀之间可以互相attend，原始文本和后缀可以互相attend）
                new_mask = torch.ones(batch_size, 8, 8, dtype=orig_masks.dtype, device=device)
                new_mask2 = torch.ones(batch_size, 8, seq_len_m, dtype=orig_masks.dtype, device=device)
                orig_mask_expanded = torch.ones(batch_size, seq_len_m, 8, dtype=orig_masks.dtype, device=device)
                text_dict['masks'] = torch.cat([
                    torch.cat([orig_masks, orig_mask_expanded], dim=-1),
                    torch.cat([new_mask2, new_mask], dim=-1)
                ], dim=-2)
        
        return text_dict

    def _init_layers(self) -> None:
        super()._init_layers()
               
    def init_weights(self):
        super().init_weights()
        self.backbone_vi.load_state_dict(self.backbone.state_dict())
        self.neck_vi.load_state_dict(self.neck.state_dict())

    def extract_feat(self, batch_inputs, batch_inputs_vi=None):
        feats_ir = self.neck(self.backbone(batch_inputs))
        if batch_inputs_vi is not None:
            feats_vi = self.neck_vi(self.backbone_vi(batch_inputs_vi))
            
            # 【修改点 2】变量名变更为 wfac_layers
            if self.wfac_layers is not None:
                feats_ir, feats_vi = self._apply_wfac_fusion(feats_ir, feats_vi)
            
            return feats_ir, feats_vi
        return feats_ir, None

    def _apply_wfac_fusion(self, feats_ir, feats_vi):
        fused_ir = []
        fused_vi = []
        # 【修改点 3】逐多尺度特征层进行独立的自适应对齐增强
        for i, (f_ir, f_vi) in enumerate(zip(feats_ir, feats_vi)):
            
            # 边界安全防御：如果下采样特征图的分辨率为奇数，小波变换会发生物理对齐崩溃
            # 在此阶段动态填充 1 像素，确保小波域操作绝对安全
            h, w = f_ir.shape[-2:]
            pad_h = 1 if h % 2 != 0 else 0
            pad_w = 1 if w % 2 != 0 else 0
            
            if pad_h > 0 or pad_w > 0:
                f_ir = torch.nn.functional.pad(f_ir, (0, pad_w, 0, pad_h), mode='replicate')
                f_vi = torch.nn.functional.pad(f_vi, (0, pad_w, 0, pad_h), mode='replicate')

            # 调用当前 Level 专属的 WFAC 映射层
            f_vi_enhanced, f_ir_enhanced, _, _ = self.wfac_layers[i](f_vi, f_ir)
            
            # 变换完成后，如果是填充过的特征图，必须精准还原切回原分辨率，否则后面的 Transformer 形状对不上
            if pad_h > 0 or pad_w > 0:
                f_ir_enhanced = f_ir_enhanced[..., :h, :w]
                f_vi_enhanced = f_vi_enhanced[..., :h, :w]

            fused_ir.append(f_ir_enhanced)
            fused_vi.append(f_vi_enhanced)
        return tuple(fused_ir), tuple(fused_vi)

    def forward(self, inputs, data_samples=None, mode='tensor', **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def _fuse_text_features(self, text_ir, text_rgb, text_dict):
        fused_text = text_ir + text_rgb
        text_dict_fused = text_dict.copy()
        text_dict_fused['embedded'] = fused_text
        return text_dict_fused
    
    def _enhance_single_modality(self, encoder_inputs_dict, text_dict, modality='ir'):
        """单模态跨模态特征增强核心逻辑，直接接收展平后的特征字典"""
        text_token_mask = text_dict['text_token_mask']
        encoder = self.encoder 

        memory, memory_text = encoder(
            query=encoder_inputs_dict['feat'],
            query_pos=encoder_inputs_dict['feat_pos'],
            key_padding_mask=encoder_inputs_dict['feat_mask'],
            spatial_shapes=encoder_inputs_dict['spatial_shapes'],
            level_start_index=encoder_inputs_dict['level_start_index'],
            valid_ratios=encoder_inputs_dict['valid_ratios'],
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])

        return memory, memory_text

    # ===================================================================== #
    # 下方所有的 loss 和 predict 相关代码完全不动，保持原样 #
    # ===================================================================== #

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             inputs2: Tensor = None,
             **kwargs) -> Union[dict, list]:
        """双模态训练损失计算"""
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_prompt:
            text_dict['embedded'] = text_dict['embedded'] + \
                self.tunable_linear.weight[:text_dict['embedded'].size(1), :].unsqueeze(0)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        
        # 拼接类别特定的可学习后缀向量
        text_dict = self._append_suffix_tokens(text_dict, batch_data_samples)

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        # 【双模态混合精度控制】
        if self.use_autocast:
            with autocast(enabled=True):
                return self._forward_loss_core(
                    batch_inputs, inputs2, text_dict, batch_data_samples)
        else:
            return self._forward_loss_core(
                batch_inputs, inputs2, text_dict, batch_data_samples)

    def _forward_loss_core(self, batch_inputs, inputs2, text_dict, batch_data_samples):
        """修改后的损失计算核心图逻辑"""
        # 1. 提取两路模态的 Backbone/Neck 特征
        feats_ir, feats_vi = self.extract_feat(batch_inputs, inputs2)

        # 2. 提前进行 pre_transformer 转换，生成各自独立的位置编码字典
        encoder_inputs_ir, decoder_inputs_dict = self.pre_transformer(feats_ir, batch_data_samples)

        # 3. 跨模态特征独立增强 (现已在内部使用共享 Encoder)
        enhanced_ir, enhanced_text_ir = self._enhance_single_modality(
            encoder_inputs_ir, text_dict, modality='ir'
        )

        if feats_vi is not None:
            encoder_inputs_vi, _ = self.pre_transformer(feats_vi, batch_data_samples)
            enhanced_rgb, enhanced_text_rgb = self._enhance_single_modality(
                encoder_inputs_vi, text_dict, modality='vi'
            )
            # 4. 增强特征中端加法融合
            fused_visual = enhanced_ir + enhanced_rgb
            text_dict_fused = self._fuse_text_features(
                enhanced_text_ir, enhanced_text_rgb, text_dict
            )
        else:
            fused_visual = enhanced_ir
            text_dict_fused = text_dict

        # 5. 构建标准的跨模态融合特征字典，接入后续的两阶段初始化与解码网络
        encoder_outputs_dict = dict(
            memory=fused_visual,
            memory_mask=encoder_inputs_ir['feat_mask'],
            spatial_shapes=encoder_inputs_ir['spatial_shapes'],
            memory_text=text_dict_fused['embedded'],
            text_token_mask=text_dict_fused['text_token_mask']
        )

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples,
                rescale: bool = True, inputs2: Tensor = None, **kwargs):
        """双模态推理预测"""
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        if self.use_autocast:
            with autocast(enabled=True):
                return self._forward_predict_core(
                    batch_inputs, inputs2, batch_data_samples, rescale,
                    token_positive_maps, text_prompts, entities)
        else:
            return self._forward_predict_core(
                batch_inputs, inputs2, batch_data_samples, rescale,
                token_positive_maps, text_prompts, entities)

    def _forward_predict_core(self, batch_inputs, inputs2, batch_data_samples,
                              rescale, token_positive_maps, text_prompts, entities):
        """修改后的推理预测核心图逻辑"""
        feats_ir, feats_vi = self.extract_feat(batch_inputs, inputs2)

        if isinstance(text_prompts[0], list):
            # 批量文本处理逻辑
            assert len(batch_inputs) == 1
            count = 0
            results_list = []
            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                if self.text_prompt:
                    text_dict['embedded'] = text_dict['embedded'] + \
                        self.tunable_linear.weight[:text_dict['embedded'].size(1), :].unsqueeze(0)
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
                
                # 拼接类别特定的可学习后缀向量
                text_dict = self._append_suffix_tokens(text_dict, batch_data_samples)

                encoder_inputs_ir, decoder_inputs_dict = self.pre_transformer(feats_ir, batch_data_samples)

                enhanced_ir, enhanced_text_ir = self._enhance_single_modality(
                    encoder_inputs_ir, text_dict, modality='ir'
                )

                if feats_vi is not None:
                    encoder_inputs_vi, _ = self.pre_transformer(feats_vi, batch_data_samples)
                    enhanced_rgb, enhanced_text_rgb = self._enhance_single_modality(
                        encoder_inputs_vi, text_dict, modality='vi'
                    )
                    fused_visual = enhanced_ir + enhanced_rgb
                    text_dict_fused = self._fuse_text_features(
                        enhanced_text_ir, enhanced_text_rgb, text_dict)
                else:
                    fused_visual = enhanced_ir
                    text_dict_fused = text_dict

                batch_data_samples[0].token_positive_map = token_positive_maps_once

                encoder_outputs_dict = dict(
                    memory=fused_visual,
                    memory_mask=encoder_inputs_ir['feat_mask'],
                    spatial_shapes=encoder_inputs_ir['spatial_shapes'],
                    memory_text=text_dict_fused['embedded'],
                    text_token_mask=text_dict_fused['text_token_mask']
                )

                tmp_dec_in, head_inputs_dict = self.pre_decoder(
                    **encoder_outputs_dict, batch_data_samples=batch_data_samples)
                decoder_inputs_dict.update(tmp_dec_in)

                decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
                head_inputs_dict.update(decoder_outputs_dict)

                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
                
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # 单文本处理逻辑
            text_dict = self.language_model(list(text_prompts))
            if self.text_prompt:
                text_dict['embedded'] = text_dict['embedded'] + \
                    self.tunable_linear.weight[:text_dict['embedded'].size(1), :].unsqueeze(0)
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
            
            # 拼接类别特定的可学习后缀向量
            text_dict = self._append_suffix_tokens(text_dict, batch_data_samples)

            encoder_inputs_ir, decoder_inputs_dict = self.pre_transformer(feats_ir, batch_data_samples)

            enhanced_ir, enhanced_text_ir = self._enhance_single_modality(
                encoder_inputs_ir, text_dict, modality='ir'
            )

            if feats_vi is not None:
                encoder_inputs_vi, _ = self.pre_transformer(feats_vi, batch_data_samples)
                enhanced_rgb, enhanced_text_rgb = self._enhance_single_modality(
                    encoder_inputs_vi, text_dict, modality='vi'
                )
                fused_visual = enhanced_ir + enhanced_rgb
                text_dict_fused = self._fuse_text_features(
                    enhanced_text_ir, enhanced_text_rgb, text_dict)
            else:
                fused_visual = enhanced_ir
                text_dict_fused = text_dict

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            encoder_outputs_dict = dict(
                memory=fused_visual,
                memory_mask=encoder_inputs_ir['feat_mask'],
                spatial_shapes=encoder_inputs_ir['spatial_shapes'],
                memory_text=text_dict_fused['embedded'],
                text_token_mask=text_dict_fused['text_token_mask']
            )

            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples)
            decoder_inputs_dict.update(tmp_dec_in)

            decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
            head_inputs_dict.update(decoder_outputs_dict)

            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        # 【完全复用父类结果后处理逻辑】
        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples