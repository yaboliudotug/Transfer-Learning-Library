"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
from typing import List, Dict
from detectron2.structures import Instances
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    Res5ROIHeads,
    StandardROIHeads,
    select_foreground_proposals,
)

from .fast_output_layers import FastRCNNOutputLayersIoU
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
import inspect
import logging
from .roi_heads import TLRes5ROIHeads
from detectron2.modeling.roi_heads.mask_head import build_mask_head
logger = logging.getLogger(__name__)

@ROI_HEADS_REGISTRY.register()
class TLRes5ROIHeadsIoU(TLRes5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    Args:
        in_features (list[str]): list of backbone feature map names to use for
            feature extraction
        pooler (ROIPooler): pooler to extra region features from backbone
        res5 (nn.Sequential): a CNN to compute per-region features, to be used by
            ``box_predictor`` and ``mask_head``. Typically this is a "res5"
            block from a ResNet.
        box_predictor (nn.Module): make box predictions from the feature.
            Should have the same interface as :class:`FastRCNNOutputLayers`.
        mask_head (nn.Module): transform features to make mask predictions

    Inputs:
        - images (ImageList):
        - features (dict[str,Tensor]): input data as a mapping from feature
          map name to tensor. Axis 0 represents the number of images `N` in
          the input data; axes 1-3 are channels, height, and width, which may
          vary between feature maps (e.g., if a feature pyramid is used).
        - proposals (list[Instances]): length `N` list of `Instances`. The i-th
          `Instances` contains object proposals for the i-th input image,
          with fields "proposal_boxes" and "objectness_logits".
        - targets (list[Instances], optional): length `N` list of `Instances`. The i-th
          `Instances` contains the ground-truth per-instance annotations
          for the i-th input image.  Specify `targets` during training only.
          It may have the following fields:
            - gt_boxes: the bounding box of each instance.
            - gt_classes: the label for each instance with a category ranging in [0, #class].
            - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
            - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        - labeled (bool, optional): whether has ground-truth label. Default: True

    Outputs:
        - list[Instances]: length `N` list of `Instances` containing the
          detected instances. Returned during inference only; may be [] during training.

        - dict[str->Tensor]:
          mapping from a named loss to a tensor storing the loss. Used during training only.
    """
    def __init__(self, *args, **kwargs):
        super(TLRes5ROIHeadsIoU, self).__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg, input_shape)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayersIoU(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        return ret

    def forward(self, images, features, proposals, targets=None, labeled=True, batched_inputs=None):
        """"""
        # print('here........')
        # print(features.keys(), features['res4'].shape)
        del images
        

        if self.training:
            if labeled:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.sample_unlabeled_proposals(proposals)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        # print(box_features.shape)
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            if labeled:
                losses = self.box_predictor.losses(predictions, proposals, batched_inputs)
                if self.mask_on:
                    proposals, fg_selection_masks = select_foreground_proposals(
                        proposals, self.num_classes
                    )
                    # Since the ROI feature transform is shared between boxes and masks,
                    # we don't need to recompute features. The mask loss is only defined
                    # on foreground proposals, so we need to select out the foreground
                    # features.
                    mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                    # del box_features
                    losses.update(self.mask_head(mask_features, proposals))
            else:
                losses = {}
            outputs = {
                'predictions': predictions[0],
                'box_features': box_features
            }
            return outputs, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    @torch.no_grad()
    def sample_unlabeled_proposals(
        self, proposals: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some unlabeled proposals.
        It returns top ``self.batch_size_per_image`` samples from proposals

        Args:
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".

        Returns:
            length `N` list of `Instances`s containing the proposals sampled for training.
        """
        return [proposal[:self.batch_size_per_image] for proposal in proposals]

