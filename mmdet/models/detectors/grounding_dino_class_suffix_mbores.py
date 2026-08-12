# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from .grounding_dino_class_suffix import GroundingDINOClassSuffix


class ProposalGraphReasoner(nn.Module):
    """Lightweight proposal graph reasoner inspired by MB-ORES.

    It fuses proposal appearance, proposal location and a pooled text context,
    then runs self-attention over the selected proposals to refine their
    features before decoder queries are formed.
    """

    def __init__(self,
                 embed_dims: int,
                 hidden_dims: Optional[int] = None,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or embed_dims
        self.input_proj = nn.Linear(embed_dims * 2 + 4, hidden_dims)
        self.input_norm = nn.LayerNorm(hidden_dims)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(
                    hidden_dims, num_heads, dropout=dropout, batch_first=True),
                norm1=nn.LayerNorm(hidden_dims),
                ffn=nn.Sequential(
                    nn.Linear(hidden_dims, hidden_dims * 4),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dims * 4, hidden_dims),
                ),
                norm2=nn.LayerNorm(hidden_dims),
            )))
        self.output_proj = nn.Linear(hidden_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                proposal_feats: Tensor,
                proposal_boxes: Tensor,
                text_context: Tensor) -> Tensor:
        text_context = text_context.unsqueeze(1).expand(
            -1, proposal_feats.size(1), -1)
        x = torch.cat([proposal_feats, proposal_boxes, text_context], dim=-1)
        x = self.input_norm(self.input_proj(x))
        for layer in self.layers:
            attn_out, _ = layer['attn'](x, x, x, need_weights=False)
            x = layer['norm1'](x + self.dropout(attn_out))
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + self.dropout(ffn_out))
        return proposal_feats + self.output_proj(x)


@MODELS.register_module()
class GroundingDINOClassSuffixMBORES(GroundingDINOClassSuffix):
    """Grounding DINO with class suffixes and MB-ORES-style proposal reasoning."""

    def __init__(self,
                 graph_hidden_dims: Optional[int] = None,
                 graph_num_heads: int = 8,
                 graph_num_layers: int = 2,
                 graph_dropout: float = 0.1,
                 *args,
                 **kwargs) -> None:
        self.graph_hidden_dims = graph_hidden_dims
        self.graph_num_heads = graph_num_heads
        self.graph_num_layers = graph_num_layers
        self.graph_dropout = graph_dropout
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        super()._init_layers()
        self.proposal_graph_reasoner = ProposalGraphReasoner(
            embed_dims=self.embed_dims,
            hidden_dims=self.graph_hidden_dims,
            num_heads=self.graph_num_heads,
            num_layers=self.graph_num_layers,
            dropout=self.graph_dropout)

    def _pool_text_context(self, text_dict: Dict) -> Tensor:
        text_embed = text_dict['embedded']
        text_mask = text_dict['text_token_mask'].float().unsqueeze(-1)
        pooled = (text_embed * text_mask).sum(dim=1)
        denom = text_mask.sum(dim=1).clamp(min=1.0)
        return pooled / denom

    def _reason_over_proposals(self, memory: Tensor, proposal_boxes: Tensor,
                               text_context: Tensor,
                               proposal_indices: Tensor) -> Tensor:
        bs, _, c = memory.shape
        gather_index = proposal_indices.unsqueeze(-1).expand(-1, -1, c)
        selected_memory = torch.gather(memory, 1, gather_index)
        selected_boxes = torch.gather(
            proposal_boxes, 1, proposal_indices.unsqueeze(-1).expand(-1, -1, 4))
        refined_selected = self.proposal_graph_reasoner(
            selected_memory, selected_boxes, text_context)
        refined_memory = memory.clone()
        refined_memory.scatter_(1, gather_index, refined_selected)
        return refined_memory

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        text_dict = dict(
            embedded=memory_text,
            text_token_mask=text_token_mask,
        )
        text_context = self._pool_text_context(text_dict)

        cls_branch = self.bbox_head.cls_branches[self.decoder.num_layers]
        reg_branch = self.bbox_head.reg_branches[self.decoder.num_layers]

        enc_outputs_class = cls_branch(output_memory, memory_text,
                                       text_token_mask)
        enc_outputs_coord_unact = reg_branch(output_memory) + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        reasoned_memory = self._reason_over_proposals(
            output_memory, enc_outputs_coord_unact.sigmoid(), text_context,
            topk_indices)

        enc_outputs_class = cls_branch(reasoned_memory, memory_text,
                                       text_token_mask)
        enc_outputs_coord_unact = reg_branch(reasoned_memory) + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        cls_out_features = cls_branch.max_text_len
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=reasoned_memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
