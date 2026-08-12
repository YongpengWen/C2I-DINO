# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .grounding_dino import GroundingDINO


@MODELS.register_module()
class GroundingDINOSpatialContrast(GroundingDINO):
    """Grounding DINO variant for spatial contrast experiments.

    It keeps the custom detector type separate from the original
    ``GroundingDINO`` implementation. Loss parsing is inherited unchanged, so
    ``loss_spatial_contrast`` is logged separately and also included in the
    optimized total loss.
    """
