import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
from matplotlib.pyplot import axis
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
# from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference

class FastRCNNOutputLayersIoU(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        use_fed_loss: bool = False,
        use_sigmoid_ce: bool = False,
        get_fed_loss_cls_weights: Optional[Callable] = None,
        fed_loss_num_classes: int = 50,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
            use_fed_loss (bool): whether to use federated loss which samples additional negative
                classes to calculate the loss
            use_sigmoid_ce (bool): whether to calculate the loss using weighted average of binary
                cross entropy with logits. This could be used together with federated loss
            get_fed_loss_cls_weights (Callable): a callable which takes dataset name and frequency
                weight power, and returns the probabilities to sample negative classes for
                federated loss. The implementation can be found in
                detectron2/data/detection_utils.py
            fed_loss_num_classes (int): number of federated classes to keep in total
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.iou_pred = nn.Linear(input_size, num_classes + 1)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.iou_pred.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.iou_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight, "loss_iou": loss_weight}
        self.loss_weight = loss_weight
        self.use_fed_loss = use_fed_loss
        self.use_sigmoid_ce = use_sigmoid_ce
        self.fed_loss_num_classes = fed_loss_num_classes

        if self.use_fed_loss:
            assert self.use_sigmoid_ce, "Please use sigmoid cross entropy loss with federated loss"
            fed_loss_cls_weights = get_fed_loss_cls_weights()
            assert (
                len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"               : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg"     : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"            : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"         : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"           : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"       : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"         : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"               : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},  # noqa
            "use_fed_loss"              : cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            "use_sigmoid_ce"            : cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            "get_fed_loss_cls_weights"  : lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER),  # noqa
            "fed_loss_num_classes"      : cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        ious = self.iou_pred(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas, ious

    def losses(self, predictions, proposals, batched_inputs=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, ious = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean")


        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
            "loss_iou": self.iou_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes, pred_iou=ious, batched_inputs=batched_inputs
            )
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/fed_loss.py  # noqa
    # with slight modifications
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: number of classes to keep in total, including both unique gt
                classes and sampled negative classes
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes

        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        return fed_loss_classes

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/custom_fast_rcnn.py#L113  # noqa
    # with slight modifications
    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """
        Args:
            pred_class_logits: shape (N, K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        N = pred_class_logits.shape[0]
        K = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(N, K + 1)
        target[range(len(gt_classes)), gt_classes] = 1
        target = target[:, :K]

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction="none"
        )

        if self.use_fed_loss:
            fed_loss_classes = self.get_fed_loss_classes(
                gt_classes,
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=K,
                weight=self.fed_loss_cls_weights,
            )
            fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
            fed_loss_classes_mask[fed_loss_classes] = 1
            fed_loss_classes_mask = fed_loss_classes_mask[:K]
            weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()
        else:
            weight = 1

        loss = torch.sum(cls_loss * weight) / N
        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def iou_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, pred_iou, batched_inputs=None): 
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        bg_inds = nonzero_tuple(gt_classes == self.num_classes)[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
            pred_iou_fg = pred_iou[fg_inds]
            pred_iou_bg = pred_iou[bg_inds]
        else:
            pred_iou =pred_iou.view(-1, self.num_classes + 1)
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
            pred_iou_fg = pred_iou[fg_inds, gt_classes[fg_inds]]
            pred_iou_bg = pred_iou[bg_inds, gt_classes[bg_inds]]

        # print('!!!!!!!!')
        # print(fg_pred_deltas.shape)
        # # print(pred_iou_bg.shape)
        # print(pred_deltas.shape)
        # print(pred_iou_fg.shape)
        # print(fg_inds.shape)
        loss_iou_fg = _dense_iou_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
            pred_iou_fg,
            batched_inputs
        )

        # pred_bg_iou = pred_iou[bg_inds]
        pred_iou_bg = pred_iou_bg.view(-1).unsqueeze(-1)
        pred_iou_bg = torch.sigmoid(pred_iou_bg)
        
        target_bg_iou = torch.zeros(pred_iou_bg.shape).type_as(pred_iou_bg)
        # loss_iou_bg = F.mse_loss(pred_iou_bg, target_bg_iou.detach(), reduction="sum")
        # loss_iou_bg = F.l1_loss(pred_iou_bg, target_bg_iou.detach(), reduction="sum")
        loss_iou_bg = F.smooth_l1_loss(pred_iou_bg, target_bg_iou.detach(), reduction="sum")
        # print('bg iou loss ....')
        # print(pred_iou_bg[:10].view(-1))
        # print(target_bg_iou[:10].view(-1))
        
        # return (loss_iou_fg + loss_iou_bg) / max(gt_classes.numel(), 1.0)  # return 0 if empty

        loss_fg = loss_iou_fg / max(fg_inds.numel(), 1.0)
        loss_bg = loss_iou_bg / max(bg_inds.numel(), 1.0)
        loss = (loss_fg + loss_bg) / 2
        return loss * 10

    def iou_loss_fgness_combine(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, pred_iou, batched_inputs=None): 
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        bg_inds = nonzero_tuple(gt_classes == self.num_classes)[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
            pred_iou_fg = pred_iou[fg_inds]
            pred_iou_bg = pred_iou[bg_inds]
        else:
            pred_iou =pred_iou.view(-1, self.num_classes + 1)
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
            pred_iou_fg = pred_iou[fg_inds, gt_classes[fg_inds]]
            pred_iou_bg = pred_iou[bg_inds, gt_classes[bg_inds]]

        # print('!!!!!!!!')
        # print(fg_pred_deltas.shape)
        # # print(pred_iou_bg.shape)
        # print(pred_deltas.shape)
        # print(pred_iou_fg.shape)
        # print(fg_inds.shape)
        loss_iou_fg = _dense_iou_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
            pred_iou_fg,
            batched_inputs
        )

        # pred_bg_iou = pred_iou[bg_inds]
        pred_iou_bg = pred_iou_bg.view(-1).unsqueeze(-1)
        pred_iou_bg = torch.sigmoid(pred_iou_bg)
        
        target_bg_iou = torch.zeros(pred_iou_bg.shape).type_as(pred_iou_bg)
        # loss_iou_bg = F.mse_loss(pred_iou_bg, target_bg_iou.detach(), reduction="sum")
        # loss_iou_bg = F.l1_loss(pred_iou_bg, target_bg_iou.detach(), reduction="sum")
        loss_iou_bg = F.smooth_l1_loss(pred_iou_bg, target_bg_iou.detach(), reduction="sum")
        # print('bg iou loss ....')
        # print(pred_iou_bg[:10].view(-1))
        # print(target_bg_iou[:10].view(-1))
        
        # return (loss_iou_fg + loss_iou_bg) / max(gt_classes.numel(), 1.0)  # return 0 if empty

        loss_fg = loss_iou_fg / max(fg_inds.numel(), 1.0)
        loss_bg = loss_iou_bg / max(bg_inds.numel(), 1.0)
        loss = (loss_fg + loss_bg) / 2
        return loss * 10

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        ious = self.predict_ious(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference_iou(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            ious
        )
        # return fast_rcnn_inference_new(
        #     boxes,
        #     scores,
        #     image_shapes,
        #     self.test_score_thresh,
        #     self.test_nms_thresh,
        #     self.test_topk_per_image,
        #     ious
        # )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas, _ = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_ious(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        _, _, ious = predictions
        # print('$$$$')
        # print(ious.shape)
        num_inst_per_image = [len(p) for p in proposals]
        ious = F.sigmoid(ious)
        # print(ious.shape)
        return ious.split(num_inst_per_image, dim=0)

# def fast_rcnn_inference_new(
#     boxes: List[torch.Tensor],
#     scores: List[torch.Tensor],
#     image_shapes: List[Tuple[int, int]],
#     score_thresh: float,
#     nms_thresh: float,
#     topk_per_image: int,
#     ious
# ):
#     print('&&&&&')
#     print(len(scores), len(boxes))
#     result_per_image = [
#         fast_rcnn_inference_single_image_new(
#             boxes_per_image, scores_per_image, ious_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
#         )
#         for scores_per_image, boxes_per_image, ious_per_image, image_shape in zip(scores, boxes, ious, image_shapes)
#     ]
#     return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_iou(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    ious
):
    result_per_image = [
        fast_rcnn_inference_single_image_iou(
            boxes_per_image, scores_per_image, ious_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, ious_per_image, image_shape in zip(scores, boxes, ious, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


# def fast_rcnn_inference_single_image_new(
#     boxes,
#     scores,
#     ious,
#     image_shape: Tuple[int, int],
#     score_thresh: float,
#     nms_thresh: float,
#     topk_per_image: int,
# ):
#     print('@@@@')
#     valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1) & torch.isfinite(ious).all(dim=1)
#     if not valid_mask.all():
#         boxes = boxes[valid_mask]
#         scores = scores[valid_mask]
#         ious = ious[valid_mask]
    
#     print(scores.shape)
#     scores = scores[:, :-1]
#     print(scores.shape)
#     print(boxes.shape)
#     num_bbox_reg_classes = boxes.shape[1] // 4
#     # Convert to Boxes to use the `clip` function ...
#     boxes = Boxes(boxes.reshape(-1, 4))
#     boxes.clip(image_shape)
#     boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

#     # 1. Filter results based on detection scores. It can make NMS more efficient
#     #    by filtering out low-confidence detections.
#     filter_mask = scores > score_thresh  # R x K
#     # R' x 2. First column contains indices of the R predictions;
#     # Second column contains indices of classes.
#     filter_inds = filter_mask.nonzero()
#     if num_bbox_reg_classes == 1:
#         boxes = boxes[filter_inds[:, 0], 0]
#     else:
#         boxes = boxes[filter_mask]
#     # print(boxes.shape)
#     scores = scores[filter_mask]
#     # print(scores.shape)
#     # print(filter_mask.shape)
#     # print(filter_mask[:5])
#     ious = ious[filter_mask.any(1)]
#     print(ious.shape)

#     # 2. Apply NMS for each class independently.
#     keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
#     if topk_per_image >= 0:
#         keep = keep[:topk_per_image]
#     boxes, scores, ious, filter_inds = boxes[keep], scores[keep], ious[keep], filter_inds[keep]
#     print('2222222222')
#     result = Instances(image_shape)
#     result.pred_boxes = Boxes(boxes)
#     result.scores = scores
#     print('3333333333')
#     result.pred_classes = filter_inds[:, 1]
#     result.iou_ness = ious
#     print('4444444444')
#     return result, filter_inds[:, 0]



def fast_rcnn_inference_single_image_iou(
    boxes,
    scores,
    ious,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1) & torch.isfinite(ious).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        ious = ious[valid_mask]
    num_classes = scores.size()[1] - 1
    filter_mask_ious = scores > score_thresh 
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    
    # ious = [ious for i in range(num_classes)]
    # ious = torch.cat(ious, axis=1)
    # print(ious.shape, filter_mask.shape)
    ious = ious[filter_mask_ious]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, ious = boxes[keep], scores[keep], filter_inds[keep], ious[keep]
    # print(boxes.shape, scores.shape, filter_inds.shape, ious.shape)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.iou_ness = ious
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


def compute_ious_0(inputs, targets):
    """compute iou and giou"""
    inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
    targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect

    ious = area_intersect / area_union.clamp(min=eps)
    
    g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
    g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
    ac_uion = g_w_intersect * g_h_intersect
    gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)

    return ious, gious

# def _dense_iou_loss(
#         anchors: List[Union[Boxes, torch.Tensor]],
#         box2box_transform: Box2BoxTransform,
#         pred_anchor_deltas: List[torch.Tensor],
#         gt_boxes: List[torch.Tensor],
#         fg_mask: torch.Tensor,
#         box_reg_loss_type="smooth_l1",
#         smooth_l1_beta=0.0,
#         pred_iou=None
#     ):

def compute_ious(boxes0: torch.Tensor, boxes1: torch.Tensor, mode='iou'):
    """ 计算一对一交并比
    Parameters
    ----------
    box0, box1: Tensor of shape `(4, )`
    gt  , pred
    """
    # 计算交集
    xy_max = torch.min(boxes0[:, 2:], boxes1[:, 2:])
    xy_min = torch.max(boxes0[:, :2], boxes1[:, :2])
    inter = torch.clamp(xy_max-xy_min, min=0)
    inter = inter[:, 0]*inter[:, 1]
    # 计算并集
    area_boxes0 = (boxes0[:, 2]-boxes0[:, 0])*(boxes0[:, 3]-boxes0[:, 1])
    area_boxes1 = (boxes1[:, 2]-boxes1[:, 0])*(boxes1[:, 3]-boxes1[:, 1])
    if mode == 'iou':
        return inter/(area_boxes0 + area_boxes1 - inter)
    if mode == 'iof':
        return inter/(area_boxes0 + area_boxes1 - inter)
    if mode == 'foregroundness':
        # print('computing foregroundness ........')
        # print(boxes0[:5])
        # print(boxes1[:5])
        gt_h, gt_w = boxes0[:, 3] - boxes0[:, 1], boxes0[:, 2] - boxes0[:, 0]
        gt_h, gt_w = torch.clamp(gt_h, 1e-7), torch.clamp(gt_w, 1e-7)
        # print(gt_h[:5], gt_w[:5])
        xmin_delta, xmax_delta = boxes1[:, 0] - boxes0[:, 0], boxes1[:, 2] - boxes0[:, 2]
        ymin_delta, ymax_delta = boxes1[:, 1] - boxes0[:, 1], boxes1[:, 3] - boxes0[:, 3]
        xmin_delta, xmax_delta = torch.abs(xmin_delta), torch.abs(xmax_delta)
        ymin_delta, ymax_delta = torch.abs(ymin_delta), torch.abs(ymax_delta)
        # print(xmin_delta[:5], xmax_delta[:5], ymin_delta[:5], ymax_delta[:5])
        xmin_delta, xmax_delta = torch.exp(-xmin_delta / gt_w), torch.exp(-xmax_delta / gt_w)
        ymin_delta, ymax_delta = torch.exp(-ymin_delta / gt_h), torch.exp(-ymax_delta / gt_h)
        # print(xmin_delta[:5], xmax_delta[:5], ymin_delta[:5], ymax_delta[:5])
        foreground_ness = xmin_delta * xmax_delta * ymin_delta * ymax_delta
        # print(foreground_ness[:5])
        # foreground_ness = torch.pow(foreground_ness, 0.25)
        # print(foreground_ness[:5])
        # ious = inter/(area_boxes0 + area_boxes1 - inter)
        # print(ious[:5])
        return foreground_ness
        
        


def _dense_iou_loss(
    anchors,
    box2box_transform,
    pred_anchor_deltas,
    gt_boxes,
    fg_mask,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
    pred_iou=None,
    batched_inputs=None
    ):
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    # print('/////////////')
    # print(anchors.shape, gt_boxes[0].shape)
    # gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
    # gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
    # # print(gt_anchor_deltas.shape)
    # pred_box_deltas = cat(pred_anchor_deltas, dim=1)[fg_mask]
    # ious, gious = compute_ious_0(pred_box_deltas, gt_anchor_deltas)
    # print(pred_box_deltas.shape, gt_anchor_deltas.shape, ious.shape)
    # print(pred_box_deltas)
    # print(gt_anchor_deltas)
    # # print(gt_boxes)
    # print(ious)
    # print(ious.shape)
    
    
    pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
    # print('.......')
    # print(pred_boxes[0].shape)
    # print(anchors.shape, len(pred_boxes))
    # print(gt_boxes[0].shape, pred_boxes[0].shape)
    # ious = compute_ious(gt_boxes[0], pred_boxes[0], mode='iof')
    # ious = compute_ious(gt_boxes[0], pred_boxes[0], mode='iou')
    ious = compute_ious(gt_boxes[0], pred_boxes[0], mode='foregroundness')
    # print(ious.shape)


    # print('*****')
    # print(ious.shape)
    # print(pred_iou.shape)
    # print(pred_iou[fg_mask].shape)

    # loss_iou = F.binary_cross_entropy_with_logits(
    #             pred_iou.view(-1), ious.view(-1).detach(),
    #             reduction="sum"
    #         )
    
    pred_iou = pred_iou.view(-1).unsqueeze(-1)
    pred_iou = torch.sigmoid(pred_iou)
    ious = ious.view(-1).unsqueeze(-1)
    # print('fg iou loss ...')
    # print(pred_iou[:10].view(-1))
    # print(ious[:10].view(-1))
    # print(pred_iou.shape, ious.shape)
    # loss_iou = F.mse_loss(pred_iou, ious.detach(), reduction="sum")
    # loss_iou = F.l1_loss(pred_iou, ious.detach(), reduction="sum")
    loss_iou = F.smooth_l1_loss(pred_iou, ious.detach(), reduction="sum")
    return loss_iou

def _dense_iou_loss_bg(
    anchors,
    box2box_transform,
    pred_anchor_deltas,
    gt_boxes,
    fg_mask,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
    pred_iou=None
    ):
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    # print('/////////////')
    # print(anchors.shape)
    gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
    gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
    # print(gt_anchor_deltas.shape)
    pred_box_deltas = cat(pred_anchor_deltas, dim=1)[fg_mask]
    # print(pred_box_deltas.shape)
    ious, gious = compute_ious(pred_box_deltas, gt_anchor_deltas)
    # print('*****')
    # print(ious.shape)
    # print(pred_iou.shape)
    # print(pred_iou[fg_mask].shape)

    # loss_iou = F.binary_cross_entropy_with_logits(
    #             pred_iou.view(-1), ious.view(-1).detach(),
    #             reduction="sum"
    #         )
    
    pred_iou = pred_iou.view(-1).unsqueeze(-1)
    pred_iou = torch.sigmoid(pred_iou)
    ious = ious.view(-1).unsqueeze(-1)
    # print(pred_iou.shape, ious.shape)
    loss_iou = F.mse_loss(pred_iou, ious.detach(), reduction="sum")
    return loss_iou

# def _dense_iou_loss_1(
#     proposal_boxes,
#     box2box_transform,
#     pred_deltas,
#     gt_boxes,
#     fg_inds,
#     pred_iou
#     # anchors,
#     # box2box_transform,
#     # pred_anchor_deltas,
#     # gt_boxes,
#     # fg_mask,
#     # box_reg_loss_type="smooth_l1",
#     # smooth_l1_beta=0.0,
#     # pred_iou=None
#     ):
#     if isinstance(anchors[0], Boxes):
#         anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
#     else:
#         anchors = cat(anchors)

    


#     print('/////////////')
#     gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
#     gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
#     print(gt_anchor_deltas.shape)
#     pred_box_deltas = cat(pred_anchor_deltas, dim=1)[fg_mask]
#     print(pred_box_deltas.shape)
#     ious, gious = compute_ious(pred_box_deltas, gt_anchor_deltas)
#     print(ious.shape)
#     print(pred_iou.shape)
#     print(pred_iou[fg_mask].shape)

#     loss_iou = F.binary_cross_entropy_with_logits(
#                 pred_iou[fg_mask].view(-1), ious.detach(),
#                 reduction="sum"
#             )
#     return loss_iou