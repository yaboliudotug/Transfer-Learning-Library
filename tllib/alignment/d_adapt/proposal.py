"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from pickle import NONE
import torch
import copy
import numpy as np
import os
import json
from typing import Optional, Callable, List
import random
import pprint
import time
import cv2
import numpy as np

from PIL import Image
import torchvision

from iopath.common.file_io import PathManager as PathManagerBase
PathManager = PathManagerBase()


import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import crop
from detectron2.structures import pairwise_iou
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.structures import Boxes

_EXIF_ORIENT = 274  # exif 'Orientation' tag
def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    print('>>>>>')
    time1 = time.time()
    with PathManager.open(file_name, "rb") as f:
        time2 = time.time()
        image = Image.open(f)
        time3 = time.time()
        image.load()

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        # image = _apply_exif_orientation(image)
        time4 = time.time()
        print(file_name)
        print('{:.3f} {:.3f} {:.3f}'.format(time2 - time1, time3 - time2, time4 - time3))
        return image
        # return convert_PIL_to_numpy(image, format)


class ProposalMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        origin_image_shape = image.shape[:2]  # h, w

        aug_input = T.AugInput(image)
        image = aug_input.image

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                obj
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, origin_image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class ProposalGenerator(DatasetEvaluator):
    """
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a ProposalGenerator to generate proposals for each inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and generate proposals results in the end (by :meth:`evaluate`).
    """
    def __init__(self, iou_threshold=(0.4, 0.5), num_classes=20, *args, **kwargs):
        super(ProposalGenerator, self).__init__(*args, **kwargs)
        self.fg_proposal_list = []
        self.bg_proposal_list = []
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

    def process_type(self, inputs, outputs, type='instances'):
        cpu_device = torch.device('cpu')
        input_instance = inputs[0]['instances'].to(cpu_device)
        output_instance = outputs[0][type].to(cpu_device)
        filename = inputs[0]['file_name']
        height = inputs[0]['height']
        width = inputs[0]['width']
        pred_boxes = output_instance.pred_boxes
        pred_scores = output_instance.scores
        pred_classes = output_instance.pred_classes
        
        pred_boxes_np = pred_boxes.tensor.numpy()
        pred_classes_np = pred_classes.numpy()
        pred_scores_np = pred_scores.numpy()

        filter_bboxs = False
        min_bbox_length = 12
        if filter_bboxs:
            pred_boxes_ls = pred_boxes_np.tolist()
            pred_classes_ls = pred_classes_np.tolist()
            pred_scores_ls = pred_scores_np.tolist()
            pred_boxes_ls_new = []
            pred_classes_ls_new = []
            pred_scores_ls_new = []
            for i in range(len(pred_boxes_ls)):
                x1, y1, x2, y2 = pred_boxes_ls[i]
                if abs(x2 - x1) < min_bbox_length or abs(y2 - y1) < min_bbox_length:
                    continue
                pred_boxes_ls_new.append([x1, y1, x2, y2])
                pred_classes_ls_new.append(pred_classes_ls[i])
                pred_scores_ls_new.append(pred_scores_ls[i])
            pred_boxes_np = np.array(pred_boxes_ls_new)
            pred_classes_np = np.array(pred_classes_ls_new)
            pred_scores_np = np.array(pred_scores_ls_new)
            
        proposal = Proposal(
            image_id=inputs[0]['image_id'],
            filename=filename,
            pred_boxes=pred_boxes_np,
            pred_classes=pred_classes_np,
            pred_scores=pred_scores_np,
            height = height,
            width = width,
            fb_set = type
        )
        pred_boxes = Boxes(pred_boxes_np)


        # proposal = Proposal(
        #     image_id=inputs[0]['image_id'],
        #     filename=filename,
        #     pred_boxes=pred_boxes.tensor.numpy(),
        #     pred_classes=pred_classes.numpy(),
        #     pred_scores=pred_scores.numpy(),
        #     height = height,
        #     width = width,
        #     fb_set = type
        # )

        if hasattr(input_instance, 'gt_boxes'):
            gt_boxes = input_instance.gt_boxes
            # assign a gt label for each pred_box
            if pred_boxes.tensor.shape[0] == 0:
                proposal.gt_fg_classes = proposal.gt_classes = proposal.gt_ious = proposal.gt_boxes = np.array([])
            elif gt_boxes.tensor.shape[0] == 0:
                proposal.gt_fg_classes = proposal.gt_classes = np.array([self.num_classes for _ in range(pred_boxes.tensor.shape[0])])
                proposal.gt_ious = np.array([0. for _ in range(pred_boxes.tensor.shape[0])])
                proposal.gt_boxes = np.array([[0, 0, 0, 0] for _ in range(pred_boxes.tensor.shape[0])])
            else:
                gt_ious, gt_classes_idx = pairwise_iou(pred_boxes, gt_boxes).max(dim=1)
                gt_classes = input_instance.gt_classes[gt_classes_idx]
                proposal.gt_fg_classes = copy.deepcopy(gt_classes.numpy())
                gt_classes[gt_ious <= self.iou_threshold[0]] = self.num_classes  # background classes
                gt_classes[(self.iou_threshold[0] < gt_ious) & (gt_ious <= self.iou_threshold[1])] = -1  # ignore
                proposal.gt_classes = gt_classes.numpy()
                proposal.gt_ious = gt_ious.numpy()
                proposal.gt_boxes = input_instance.gt_boxes[gt_classes_idx].tensor.numpy()

                # proposal.pred_boxes = input_instance.gt_boxes[gt_classes_idx].tensor.numpy()
                # proposal.pred_classes = gt_classes.numpy()

            if gt_boxes.tensor.shape[0] == 0:
                proposal.all_gt_classes = proposal.all_gt_boxes = np.array([])
            else:
                proposal.all_gt_classes = input_instance.gt_classes.numpy()
                proposal.all_gt_boxes = input_instance.gt_boxes.tensor.numpy()
        if pred_boxes.tensor.shape[0] == 0:
            proposal.pred_ids = np.array([])
        else:
            one_pred_ids = np.array(list(range(pred_boxes.tensor.shape[0])))
            proposal.pred_ids = one_pred_ids

        return proposal

    def process(self, inputs, outputs):
        self.fg_proposal_list.append(self.process_type(inputs, outputs, "instances"))
        self.bg_proposal_list.append(self.process_type(inputs, outputs, "background"))

    def evaluate(self):
        return self.fg_proposal_list, self.bg_proposal_list

def update_proposal000(proposals, num_classes, iou_threshold=[0.03, 0.3]):
    prop_new = PersistentProposalList()
    for proposal in proposals:
        pred_boxes_np, all_gt_boxes_np, all_gt_classes_np = proposal.pred_boxes, proposal.all_gt_boxes, proposal.all_gt_classes
        # if pred_boxes_np.shape[0] == 0:
        pred_boxes, all_gt_boxes = Boxes(pred_boxes_np), Boxes(all_gt_boxes_np)
        # print('>>>>>>')
        # print(all_gt_classes_np)
        gt_ious, gt_classes_idx = pairwise_iou(pred_boxes, all_gt_boxes).max(dim=1)
        gt_classes = all_gt_classes_np[gt_classes_idx]
        if gt_classes_idx.shape[0] == 1:
            # print(gt_classes)
            gt_classes = np.array([gt_classes])
            # print(gt_classes)
        # print(gt_ious)
        # print(gt_classes_idx)
        # print(gt_classes)
        proposal.gt_fg_classes = copy.deepcopy(gt_classes)
        # print(gt_ious <= iou_threshold[0])
        # print(gt_classes)
        # print(gt_classes.shape, gt_classes_idx.shape)
        if gt_classes_idx.shape[0] == 1:
            gt_classes[0] = num_classes
        else:
            gt_classes[gt_ious <= iou_threshold[0]] = num_classes  # background classes
        # print('>>>>>>>>')
        # print(gt_classes)
        # print(type(gt_classes))
        # print(a)
        # print(gt_classes[a])
        if gt_classes_idx.shape[0] == 1:
            if (iou_threshold[0] < gt_ious) & (gt_ious <= iou_threshold[1])[0]:
                gt_classes[0] = -1
        else:
            gt_classes[(iou_threshold[0] < gt_ious) & (gt_ious <= iou_threshold[1])] = -1  # ignore
        proposal.gt_classes = gt_classes
        proposal.gt_ious = gt_ious.numpy()
        proposal.gt_boxes = all_gt_boxes[gt_classes_idx].tensor.numpy()
        prop_new.append(proposal)
    return prop_new

def update_proposal(proposals, cache_file, num_classes, iou_threshold=[0.03, 0.3]):
    print('here is the update proposal !!!!!!!!')
    prop_new = PersistentProposalList(cache_file)
    for proposal_ori in proposals:
        # print('>>>>>>>>')
        # print(proposal.gt_ious)
        proposal = copy.deepcopy(proposal_ori)

        pred_boxes_np, all_gt_boxes_np, all_gt_classes_np = proposal.pred_boxes, proposal.all_gt_boxes, proposal.all_gt_classes
        pred_boxes_Boxes, all_gt_boxes_Boxes, all_gt_classes = Boxes(pred_boxes_np), Boxes(all_gt_boxes_np), torch.from_numpy(all_gt_classes_np)
        if pred_boxes_Boxes.tensor.shape[0] == 0 or all_gt_boxes_Boxes.tensor.shape[0] == 0:
            prop_new.append(proposal)
            continue
        gt_ious, gt_classes_idx = pairwise_iou(pred_boxes_Boxes, all_gt_boxes_Boxes).max(dim=1)
        gt_classes = all_gt_classes[gt_classes_idx]
        proposal.gt_fg_classes = copy.deepcopy(gt_classes.numpy())
        gt_classes[gt_ious <= iou_threshold[0]] = num_classes  # background classes
        gt_classes[(iou_threshold[0] < gt_ious) & (gt_ious <= iou_threshold[1])] = -1  # ignore
        
        proposal.gt_classes = gt_classes.numpy()
        proposal.gt_ious = gt_ious.numpy()
        proposal.gt_boxes = all_gt_boxes_Boxes[gt_classes_idx].tensor.numpy()

        # proposal.pred_boxes = proposal.gt_boxes
        # proposal.pred_classes = proposal.gt_classes
        # proposal = proposal[proposal.pred_classes != -1]

        
        
        prop_new.append(proposal)
        # print(proposal.gt_ious)
    prop_new.flush()

        
    return prop_new

def update_proposal_9(proposals, num_classes, iou_threshold=[0.03, 0.3], remove_bg=False):
    prop_new = PersistentProposalList()
    print('update proposal gt ...')
    for proposal_ori in proposals:
        proposal = copy.deepcopy(proposal_ori)

        proposal.pred_boxes = proposal.gt_boxes
        proposal.pred_classes = proposal.gt_classes
        proposal = proposal[proposal.pred_classes != -1]
        # if not remove_bg:
        #     keep_indices = (0 <= proposal.pred_classes) & (proposal.pred_classes < num_classes)
        #     prop_new.append(proposal[~keep_indices])
        # else:
        #     prop_new.append(proposal)
        prop_new.append(proposal)
    return prop_new

def update_proposal_1(proposals, num_classes, iou_threshold=[0.03, 0.3]):
    prop_new = PersistentProposalList()
    for proposal_ori in proposals:
        proposal = copy.deepcopy(proposal_ori)
        pred_boxes_np, all_gt_boxes_np, all_gt_classes_np = proposal.pred_boxes, proposal.all_gt_boxes, proposal.all_gt_classes
        gt_fg_classes_0, gt_classes_0, gt_ious_0, gt_boxes_0 = proposal.gt_fg_classes, proposal.gt_classes, proposal.gt_ious, proposal.gt_boxes
        pred_boxes_Boxes, all_gt_boxes_Boxes, all_gt_classes = Boxes(pred_boxes_np), Boxes(all_gt_boxes_np), torch.from_numpy(all_gt_classes_np)
        if pred_boxes_Boxes.tensor.shape[0] == 0 or all_gt_boxes_Boxes.tensor.shape[0] == 0:
            prop_new.append(proposal)
            continue
        gt_ious, gt_classes_idx = pairwise_iou(pred_boxes_Boxes, all_gt_boxes_Boxes).max(dim=1)
        gt_classes = all_gt_classes[gt_classes_idx]
        proposal.gt_fg_classes = copy.deepcopy(gt_classes.numpy())
        gt_classes[gt_ious <= iou_threshold[0]] = num_classes  # background classes
        gt_classes[(iou_threshold[0] < gt_ious) & (gt_ious <= iou_threshold[1])] = -1  # ignore
        proposal.gt_classes = gt_classes.numpy()
        proposal.gt_ious = gt_ious.numpy()
        proposal.gt_boxes = all_gt_boxes_Boxes[gt_classes_idx].tensor.numpy()
        prop_new.append(proposal)

        pred_boxes_np_1, gt_fg_classes, gt_classes, gt_ious, gt_boxes = proposal.pred_boxes, proposal.gt_fg_classes, proposal.gt_classes, proposal.gt_ious, proposal.gt_boxes

        print('>>>>>>>>>')
        # print(pred_boxes_np_1 == pred_boxes_np)
        print(gt_fg_classes == gt_fg_classes_0)
        print(gt_classes == gt_classes_0)
        print(gt_ious == gt_ious_0)
        # print(gt_boxes == gt_boxes_0)

    return prop_new


class Proposal:
    """
    A data structure that stores the proposals for a single image.

    Args:
        image_id (str): unique image identifier
        filename (str): image filename
        pred_boxes (numpy.ndarray): predicted boxes
        pred_classes (numpy.ndarray): predicted classes
        pred_scores (numpy.ndarray): class confidence score
        gt_classes (numpy.ndarray, optional): ground-truth classes, including background classes
        gt_boxes (numpy.ndarray, optional): ground-truth boxes
        gt_ious (numpy.ndarray, optional): IoU between predicted boxes and ground-truth boxes
        gt_fg_classes (numpy.ndarray, optional): ground-truth foreground classes, not including background classes

    """
    def __init__(self, image_id, filename, pred_boxes, pred_classes, pred_scores,
                 gt_classes=None, gt_boxes=None, gt_ious=None, gt_fg_classes=None, 
                 all_gt_classes=None, all_gt_boxes=None, pred_ids=None, height=None, width=None, fb_set=None):
        self.image_id = image_id
        self.filename = filename
        self.pred_boxes = pred_boxes
        self.pred_classes = pred_classes
        self.pred_scores = pred_scores
        self.gt_classes = gt_classes
        self.gt_boxes = gt_boxes
        self.gt_ious = gt_ious
        self.gt_fg_classes = gt_fg_classes
        self.all_gt_classes = all_gt_classes
        self.all_gt_boxes = all_gt_boxes
        self.pred_ids = pred_ids
        self.height = height
        self.width = width
        self.fb_set = fb_set
    def to_dict(self):
        return {
            "__proposal__": True,
            "image_id": self.image_id,
            "filename": self.filename,
            "pred_boxes": self.pred_boxes.tolist(),
            "pred_classes": self.pred_classes.tolist(),
            "pred_scores": self.pred_scores.tolist(),
            "gt_classes": self.gt_classes.tolist(),
            "gt_boxes": self.gt_boxes.tolist(),
            "gt_ious": self.gt_ious.tolist(),
            "gt_fg_classes": self.gt_fg_classes.tolist(),
            "all_gt_boxes": self.all_gt_boxes.tolist(),
            "all_gt_classes": self.all_gt_classes.tolist(),
            "pred_ids": self.pred_ids.tolist(),
            "height": self.height,
            "width": self.width,
            "fb_set": self.fb_set
        }
    def keys(self):
        return [
            # "__proposal__",
            "image_id",
            "filename",
            "pred_boxes",
            "pred_classes",
            "pred_scores",
            "gt_classes",
            "gt_boxes",
            "gt_ious",
            "gt_fg_classes",
            "all_gt_boxes",
            "all_gt_classes",
            "pred_ids",
            "height",
            "width",
            "fb_set",
        ]

    def __str__(self):
        pp = pprint.PrettyPrinter(indent=2)
        return pp.pformat(self.to_dict())

    def __len__(self):
        return len(self.pred_boxes)

    def __getitem__(self, item):
        # print(self.pred_ids)
        return Proposal(
            image_id=self.image_id,
            filename=self.filename,
            pred_boxes=self.pred_boxes[item],
            pred_classes=self.pred_classes[item],
            pred_scores=self.pred_scores[item],
            gt_classes=self.gt_classes[item],
            gt_boxes=self.gt_boxes[item],
            gt_ious=self.gt_ious[item],
            gt_fg_classes=self.gt_fg_classes[item],
            all_gt_boxes=self.all_gt_boxes,
            all_gt_classes=self.all_gt_classes,
            pred_ids=self.pred_ids[item],
            height=self.height,
            width=self.width,
            fb_set=self.fb_set
        )

    def extend_0(self, proposal):
        assert self.image_id == proposal.image_id, 'image_id is not the same: {}, {}'.format(self.image_id, proposal.image_id)
        assert self.filename == proposal.filename, 'filename is not the same: {}, {}'.format(self.filename, proposal.filename)
        # self.image_id = proposal.image_id
        # self.filename = proposal.filename
        # if len(proposal.pred_boxes) and len(proposal.pred_boxes)
        self.pred_boxes = np.concatenate(self.pred_boxes, proposal.pred_boxes) if len(proposal.pred_boxes) else self.pred_boxes
        self.pred_classes = np.concatenate(self.pred_classes, proposal.pred_classes) if len(proposal.pred_classes) else self.pred_classes
        self.pred_scores = np.concatenate(self.pred_scores, proposal.pred_scores) if len(proposal.pred_scores) else self.pred_scores
        self.gt_classes = np.concatenate(self.gt_classes, proposal.gt_classes) if len(proposal.gt_classes) else self.gt_classes
        self.gt_boxes = np.concatenate(self.gt_boxes, proposal.gt_boxes) if len(proposal.gt_boxes) else self.gt_boxes
        self.gt_ious = np.concatenate(self.gt_ious, proposal.gt_ious) if len(proposal.gt_ious) else self.gt_ious
        self.gt_fg_classes = np.concatenate(self.gt_fg_classes, proposal.gt_fg_classes) if len(proposal.gt_fg_classes) else self.gt_fg_classes
        # self.all_gt_classes = proposal.all_gt_classes
        # self.all_gt_boxes = proposal.all_gt_boxes
        self.pred_ids = np.concatenate(self.pred_ids, proposal.pred_ids) if len(proposal.pred_ids) else self.pred_ids
        # self.height = proposal.height
        # self.width = proposal.width
        # self.fb_set = proposal.fb_set
        return self

    def extend(self, proposal):
        assert self.image_id == proposal.image_id, 'image_id is not the same: {}, {}'.format(self.image_id, proposal.image_id)
        assert self.filename == proposal.filename, 'filename is not the same: {}, {}'.format(self.filename, proposal.filename)
        # self.image_id = proposal.image_id
        # self.filename = proposal.filename
        if len(self.pred_boxes) and len(proposal.pred_boxes):
            # print(self.pred_boxes)
            # print(proposal.pred_boxes)
            self.pred_boxes = np.concatenate([self.pred_boxes, proposal.pred_boxes]) 
        elif len(proposal.pred_boxes):
            self.pred_boxes =  proposal.pred_boxes
        
        if len(self.pred_classes) and len(proposal.pred_classes):
            self.pred_classes = np.concatenate([self.pred_classes, proposal.pred_classes])
        elif len(proposal.pred_classes):
            self.pred_classes = proposal.pred_classes

        if len(self.pred_scores) and len(proposal.pred_scores):
            self.pred_scores = np.concatenate([self.pred_scores, proposal.pred_scores])
        elif len(proposal.pred_scores):
            self.pred_scores = proposal.pred_scores

        if len(self.gt_classes) and len(proposal.gt_classes):
            self.gt_classes = np.concatenate([self.gt_classes, proposal.gt_classes])
        elif len(proposal.gt_classes):
            self.gt_classes = proposal.gt_classes

        if len(self.gt_boxes) and len(proposal.gt_boxes):
            self.gt_boxes = np.concatenate([self.gt_boxes, proposal.gt_boxes])
        elif len(proposal.gt_boxes):
            self.gt_boxes = proposal.gt_boxes

        if len(self.gt_ious) and len(proposal.gt_ious):
            self.gt_ious = np.concatenate([self.gt_ious, proposal.gt_ious]) 
        elif len(proposal.gt_ious):
            self.gt_ious = proposal.gt_ious

        if len(self.gt_fg_classes) and len(proposal.gt_fg_classes):
            self.gt_fg_classes = np.concatenate([self.gt_fg_classes, proposal.gt_fg_classes]) 
        elif len(proposal.gt_fg_classes):
            self.gt_fg_classes = proposal.gt_fg_classes

        if len(self.pred_ids) and len(proposal.pred_ids):
            self.pred_ids = np.concatenate([self.pred_ids, proposal.pred_ids]) 
        elif len(proposal.pred_ids):
            self.pred_ids = proposal.pred_ids
        # return self



class ProposalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Proposal):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


def asProposal(dict):
    if '__proposal__' in dict:
        # print('********')
        # print(dict["image_id"])
        # print(dict.keys())
        # print(dict.keys())
        return Proposal(
            dict["image_id"],
            dict["filename"],
            np.array(dict["pred_boxes"]),
            np.array(dict["pred_classes"]),
            np.array(dict["pred_scores"]),
            np.array(dict["gt_classes"]),
            np.array(dict["gt_boxes"]),
            np.array(dict["gt_ious"]),
            np.array(dict["gt_fg_classes"]),
            np.array(dict["all_gt_classes"]),
            np.array(dict["all_gt_boxes"]),
            pred_ids=np.array(dict["pred_ids"]),
            height=dict["height"],
            width=dict["width"],
            fb_set=dict["fb_set"]
        )
    return dict


class PersistentProposalList(list):
    """
    A data structure that stores the proposals for a dataset.

    Args:
        filename (str, optional): filename indicating where to cache
    """
    def __init__(self, filename=None):
        super(PersistentProposalList, self).__init__()
        self.filename = filename

    def load(self):
        """
        Load from cache.

        Return:
            whether succeed
        """
        if os.path.exists(self.filename):
            print("Reading from cache: {}".format(self.filename))
            with open(self.filename, "r") as f:
                self.extend(json.load(f, object_hook=asProposal))
            return True
        else:
            return False

    def flush(self):
        """
        Flush to cache.
        """
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, "w") as f:
            json.dump(self, f, cls=ProposalEncoder, indent=4)
        print("Write to cache: {}".format(self.filename))


def flatten(proposal_list, max_number=100000):
    """
    Flatten a list of proposals

    Args:
        proposal_list (list):  a list of proposals grouped by images
        max_number (int): maximum number of kept proposals for each image

    """
    flattened_list = []
    for proposals in proposal_list:
        for i in range(min(len(proposals), max_number)):
            flattened_list.append(proposals[i:i+1])
    return flattened_list


class ProposalDataset(datasets.VisionDataset):
    """
    A dataset for proposals.

    Args:
        proposal_list (list): list of Proposal
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        crop_func: (ExpandCrop, optional):
    """
    def __init__(self, proposal_list: List[Proposal], transform: Optional[Callable] = None, crop_func=None, crop_img_dir=None):
        super(ProposalDataset, self).__init__("", transform=transform)
        self.proposal_list = list(filter(lambda p: len(p) > 0, proposal_list))  # remove images without proposals
        self.loader = default_loader
        self.crop_func = crop_func
        self.crop_img_dir = crop_img_dir

    def __getitem__(self, index: int):
        # get proposals for the index-th image
        proposals = self.proposal_list[index]
        # print(len(proposals))
        proposal = proposals[random.randint(0, len(proposals)-1)]
        # print(proposal)
        crop_loaded_flag = False
        image_width, image_height = int(proposal.width), int(proposal.height)
        time1 = time.time()
        time_start_read = time.time()
        read_mode = ''
        test_dir = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_adapt/test_imgs'

        img_crop, img = None, None
        # self.crop_img_dir = None
        # print(self.crop_img_dir)
        if self.crop_img_dir is not None:
            pred_id = proposal.pred_ids
            fb_set = proposal.fb_set
            crop_img_name = os.path.basename(proposals.filename).split('.')[0] + '_{}_proposal_{}.jpg'.format(fb_set, pred_id)
            crop_img_path = os.path.join(self.crop_img_dir, crop_img_name)
            if os.path.exists(crop_img_path):
                img = self.loader(crop_img_path)
                crop_loaded_flag = True
            read_mode = 'crop'
            # print('>>>>>>>>>>')
            # print(crop_img_path)
            # img.save(os.path.join(test_dir, crop_img_name.split('.jpg')[0] + '_crop.jpg'))

        if not crop_loaded_flag:
        # if True:
        # if False:
            # print('whole load:{}'.format(os.path.basename(proposals.filename)))
            img = self.loader(proposals.filename)
            # small_img_path = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/crop_img.jpg'
            # img = self.loader(small_img_path)

            time2 = time.time()

            # random sample a proposal
            
            # image_width, image_height = img.width, img.height
            # proposal_dict = proposal.to_dict()
            # proposal_dict.update(width=img.width, height=img.height)

            # crop the proposal from the whole image
            x1, y1, x2, y2 = proposal.pred_boxes
            top, left, height, width = int(y1), int(x1), int(y2 - y1), int(x2 - x1)
            if self.crop_func is not None:
                top, left, height, width = self.crop_func(img, top, left, height, width)
            img = crop(img, top, left, height, width)
            time3 = time.time()
            read_mode = 'whole'
            # img.save(os.path.join(test_dir, crop_img_name.split('.jpg')[0] + '_whole.jpg'))
            # print('proposal dataset time, read img:{:.3f}, crop img:{:.3f}'.format(time2 - time1, time3 - time2))
        time_end_read = time.time()
        # print(crop_loaded_flag, time_end_read - time1)
        # print('proposal dataset time, read img:{:.3f}, read_mode:{}'.format(time_end_read - time_start_read, read_mode))

        # if img_crop and img:
        #     print(np.array(img_crop).shape, np.array(img).shape)
        #     print(np.array(img_crop) - np.array(img))


        if self.transform is not None:
            img = self.transform(img)

        return img, {
            "image_id": proposal.image_id,
            "filename": proposal.filename,
            "pred_boxes": proposal.pred_boxes.astype(np.float),
            "pred_classes": proposal.pred_classes.astype(np.long),
            "pred_scores": proposal.pred_scores.astype(np.float),
            "gt_classes": proposal.gt_classes.astype(np.long),
            "gt_boxes": proposal.gt_boxes.astype(np.float),
            "gt_ious": proposal.gt_ious.astype(np.float),
            "gt_fg_classes": proposal.gt_fg_classes.astype(np.long),
            "width": image_width,
            "height": image_height
        }

    def __len__(self):
        return len(self.proposal_list)


class ExpandCrop:
    """
    The input of the bounding box adaptor (the crops of objects) will be larger than the original
    predicted box, so that the bounding box adapter could access more location information.
    """
    def __init__(self, expand=1.):
        self.expand = expand

    def __call__(self, img, top, left, height, width):
        cx = left + width / 2.
        cy = top + height / 2.
        height = round(height * self.expand)
        width = round(width * self.expand)
        new_top = round(cy - height / 2.)
        new_left = round(cx - width / 2.)
        return new_top, new_left, height, width