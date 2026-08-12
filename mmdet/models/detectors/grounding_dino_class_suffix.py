import re
from typing import Dict, Sequence

import torch
import torch.nn as nn
from mmengine.logging import MMLogger

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from .grounding_dino import GroundingDINO


@MODELS.register_module()
class GroundingDINOClassSuffix(GroundingDINO):
    """Grounding DINO with class/direction learnable text suffix tokens.

    Suffix tokens are appended after language features are projected to the
    detector embedding dimension. They are auxiliary context only; positive maps
    still supervise the original target phrase spans from ``tokens_positive``.
    """

    def __init__(self,
                 class_names: Sequence[str],
                 direction_names: Sequence[str] = (),
                 suffix_len: int = 8,
                 direction_suffix_len: int = 0,
                 suffix_init_std: float = 0.01,
                 *args,
                 **kwargs) -> None:
        self.class_names = tuple(class_names)
        self.class_to_idx = {
                class_name.lower(): idx
                for idx, class_name in enumerate(self.class_names)
        }
        self.direction_names = tuple(direction_names)
        self.direction_to_idx = {
            direction_name.lower(): idx
            for idx, direction_name in enumerate(self.direction_names)
        }
        self.suffix_len = suffix_len
        self.direction_suffix_len = direction_suffix_len
        self.suffix_init_std = suffix_init_std
        self._suffix_capacity_warning_emitted = False
        self._suffix_append_logged = False
        self.direction_aliases = (
            ('upper_left',
             ('upper left', 'top left', 'northwest', 'north-west',
              'northwestern', 'northwesternmost')),
            ('upper_right',
             ('upper right', 'top right', 'northeast', 'north-east',
              'northeastern', 'northeasternmost')),
            ('lower_left',
             ('lower left', 'bottom left', 'southwest', 'south-west',
              'southwestern', 'southwesternmost')),
            ('lower_right',
             ('lower right', 'bottom right', 'southeast', 'south-east',
              'southeastern', 'southeasternmost')),
            ('left', ('left', 'west', 'western', 'westmost', 'westernmost',
                      'leftmost')),
            ('right', ('right', 'east', 'eastern', 'eastmost', 'easternmost',
                       'rightmost')),
            ('upper', ('upper', 'top', 'above', 'north', 'northern',
                       'northmost', 'northernmost', 'topmost')),
            ('lower', ('lower', 'bottom', 'below', 'south', 'southern',
                       'southmost', 'southernmost', 'bottommost')),
            ('middle', ('middle', 'center', 'central', 'centre')),
        )
        direction_phrase = '|'.join(
            re.escape(alias)
            for _, aliases in self.direction_aliases
            for alias in aliases)
        self.direction_patterns = [
            (direction_name,
             re.compile(r'\b(' + '|'.join(
                 re.escape(alias) for alias in aliases) + r')\b'))
            for direction_name, aliases in self.direction_aliases
        ]
        self.direction_relation_pattern = re.compile(
            r'\b(' + direction_phrase + r')\b\s+of\b')
        self.direction_inverse = {
            'left': 'right',
            'right': 'left',
            'upper': 'lower',
            'lower': 'upper',
            'upper_left': 'lower_right',
            'upper_right': 'lower_left',
            'lower_left': 'upper_right',
            'lower_right': 'upper_left',
            'middle': 'middle',
            'unknown': 'unknown',
        }
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        super()._init_layers()
        self.class_suffix_tokens = nn.Parameter(
            torch.randn(len(self.class_names), self.suffix_len,
                        self.embed_dims) * self.suffix_init_std)
        if self.direction_names and self.direction_suffix_len > 0:
            self.direction_suffix_tokens = nn.Parameter(
                torch.randn(len(self.direction_names),
                            self.direction_suffix_len,
                            self.embed_dims) * self.suffix_init_std)

    def _get_class_suffix_indices(self, batch_data_samples: SampleList,
                                  device: torch.device) -> torch.Tensor:
        indices = []
        for data_sample in batch_data_samples:
            target_class = data_sample.get('target_class', None)
            if target_class is None:
                indices.append(0)
                continue
            indices.append(self.class_to_idx.get(str(target_class).lower(), 0))
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _normalize_direction_alias(self, direction_text: str) -> str:
        direction_text = str(direction_text).lower()
        for direction_name, aliases in self.direction_aliases:
            if direction_text in aliases:
                return direction_name
        return 'unknown'

    def _find_first_direction(self, text: str) -> str:
        matches = []
        for direction_name, pattern in self.direction_patterns:
            match = pattern.search(text)
            if match is not None:
                matches.append((match.start(), -len(match.group(0)),
                                direction_name))
        if not matches:
            return 'unknown'
        matches.sort()
        return matches[0][2]

    def _extract_direction_name(self,
                                text: str,
                                target_class: str = None) -> str:
        text = str(text).lower()
        first_sentence = re.split(r'[.!?]', text, maxsplit=1)[0]

        direction_name = self._find_first_direction(first_sentence)
        if direction_name != 'unknown':
            return direction_name

        target_class = str(target_class or '').lower()
        if target_class:
            escaped_target = re.escape(target_class)
            direct_target = re.search(
                escaped_target + r'.{0,120}?' +
                self.direction_relation_pattern.pattern, text)
            if direct_target is not None:
                return self._normalize_direction_alias(direct_target.group(1))

            inverse_reference = re.search(
                self.direction_relation_pattern.pattern +
                r'.{0,40}?\b(?:a|an|the)?\s*' + escaped_target + r'\b',
                text)
            if inverse_reference is not None:
                ref_direction = self._normalize_direction_alias(
                    inverse_reference.group(1))
                return self.direction_inverse.get(ref_direction, ref_direction)

        return self._find_first_direction(text)

    def _get_direction_suffix_indices(self, batch_data_samples: SampleList,
                                      device: torch.device) -> torch.Tensor:
        unknown_idx = self.direction_to_idx.get('unknown', 0)
        indices = []
        for data_sample in batch_data_samples:
            direction_name = data_sample.get('target_direction', None)
            if direction_name is None:
                direction_name = self._extract_direction_name(
                    data_sample.get('text', ''),
                    data_sample.get('target_class', None))
            indices.append(
                self.direction_to_idx.get(str(direction_name).lower(),
                                          unknown_idx))
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _append_suffix_embeddings(self, text_dict: Dict,
                                  suffix: torch.Tensor) -> Dict:
        embeddings = text_dict['embedded']
        batch_size, seq_len, _ = embeddings.shape
        append_len = suffix.size(1)
        text_dict['embedded'] = torch.cat([embeddings, suffix], dim=1)

        token_mask = text_dict['text_token_mask']
        suffix_token_mask = torch.ones(
            batch_size, append_len, dtype=token_mask.dtype,
            device=embeddings.device)
        text_dict['text_token_mask'] = torch.cat(
            [token_mask, suffix_token_mask], dim=1)

        if text_dict.get('position_ids', None) is not None:
            new_seq_len = seq_len + append_len
            text_dict['position_ids'] = torch.arange(
                new_seq_len, dtype=torch.long,
                device=embeddings.device).unsqueeze(0).expand(batch_size, -1)

        if text_dict.get('masks', None) is not None:
            masks = text_dict['masks']
            if masks.dim() == 2:
                suffix_mask = torch.ones(
                    batch_size, append_len, dtype=masks.dtype,
                    device=embeddings.device)
                text_dict['masks'] = torch.cat([masks, suffix_mask], dim=1)
            elif masks.dim() == 3:
                suffix_cols = torch.ones(
                    batch_size,
                    seq_len,
                    append_len,
                    dtype=masks.dtype,
                    device=embeddings.device)
                suffix_rows = torch.ones(
                    batch_size,
                    append_len,
                    seq_len + append_len,
                    dtype=masks.dtype,
                    device=embeddings.device)
                text_dict['masks'] = torch.cat([
                    torch.cat([masks, suffix_cols], dim=-1), suffix_rows
                ],
                                                dim=-2)
        return text_dict

    def _append_class_suffix(self, text_dict: Dict,
                             batch_data_samples: SampleList) -> Dict:
        if self.suffix_len <= 0 and self.direction_suffix_len <= 0:
            return text_dict

        text_dict = text_dict.copy()
        embeddings = text_dict['embedded']
        max_text_len = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        remain_len = max_text_len - embeddings.size(1)
        if remain_len <= 0:
            if not self._suffix_capacity_warning_emitted:
                MMLogger.get_current_instance().warning(
                    'No class/direction suffix tokens were appended because '
                    'the text sequence already occupies the classifier limit '
                    '(%d tokens). Set language_model.pad_to_max=False or '
                    'reserve capacity by lowering language_model.max_tokens.',
                    max_text_len)
                self._suffix_capacity_warning_emitted = True
            return text_dict

        device = embeddings.device
        class_append_len = min(self.suffix_len, remain_len)
        if class_append_len > 0:
            suffix_indices = self._get_class_suffix_indices(
                batch_data_samples, device)
            suffix = self.class_suffix_tokens[
                suffix_indices, :class_append_len]
            text_dict = self._append_suffix_embeddings(text_dict, suffix)
            remain_len -= class_append_len

        direction_append_len = min(self.direction_suffix_len, remain_len)
        if (direction_append_len > 0 and self.direction_names
                and hasattr(self, 'direction_suffix_tokens')):
            direction_indices = self._get_direction_suffix_indices(
                batch_data_samples, device)
            suffix = self.direction_suffix_tokens[
                direction_indices, :direction_append_len]
            text_dict = self._append_suffix_embeddings(text_dict, suffix)
        if not self._suffix_append_logged:
            appended_len = text_dict['embedded'].size(1) - embeddings.size(1)
            MMLogger.get_current_instance().info(
                'Appended %d class/direction suffix tokens to text sequences '
                'of length %d (classifier limit: %d).', appended_len,
                embeddings.size(1), max_text_len)
            self._suffix_append_logged = True
        return text_dict

    def _encode_text(self, text_prompts, batch_data_samples):
        text_dict = self.language_model(text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        return self._append_class_suffix(text_dict, batch_data_samples)

    def loss(self, batch_inputs, batch_data_samples):
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
                    self.get_tokens_and_prompts(text_prompts[0], True)
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
                        self.get_tokens_and_prompts(text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self._encode_text(new_text_prompts, batch_data_samples)

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        if self.use_autocast:
            from mmengine.runner.amp import autocast
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            enhanced_text_prompts.append(data_samples.get(
                'caption_prompt', None))
            tokens_positives.append(data_samples.get('tokens_positive', None))

        custom_entities = batch_data_samples[0].get('custom_entities', False)
        if len(text_prompts) == 1:
            maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities,
                    enhanced_text_prompts[0], tokens_positives[0])
            ] * len(batch_inputs)
        else:
            maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompt, custom_entities, enhanced_text_prompt,
                    tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(*maps_and_prompts)

        visual_feats = self.extract_feat(batch_inputs)
        if isinstance(text_prompts[0], list):
            return super().predict(batch_inputs, batch_data_samples, rescale)

        text_dict = self._encode_text(list(text_prompts), batch_data_samples)

        is_rec_tasks = []
        for i, data_samples in enumerate(batch_data_samples):
            is_rec_tasks.append(token_positive_maps[i] is None)
            data_samples.token_positive_map = token_positive_maps[i]

        head_inputs_dict = self.forward_transformer(
            visual_feats, text_dict, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                    elif labels >= len(entity):
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
