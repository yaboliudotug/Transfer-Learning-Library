from genericpath import exists
import logging
import os
import argparse
from pickle import NONE
from random import random, shuffle
import sys
import pprint
from turtle import clone
import numpy as np
import cv2
from regex import L
import tqdm
import copy
# import copy

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import default_writers, launch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
import detectron2.utils.comm as comm
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    MetadataCatalog
)
from detectron2.utils.events import EventStorage
from detectron2.evaluation import inference_on_dataset
from detectron2.structures import pairwise_iou
from detectron2.structures import Boxes

sys.path.append('../../../..')
import tllib.alignment.d_adapt.modeling.meta_arch as models
from tllib.alignment.d_adapt.proposal import ProposalGenerator, ProposalMapper, PersistentProposalList, flatten, update_proposal
from tllib.alignment.d_adapt.feedback import get_detection_dataset_dicts, DatasetMapper

sys.path.append('..')
import utils
from utils import PascalVOCDetectionPerClassEvaluatorMAP

import category_adaptation
import bbox_adaptation

from detectron2.layers import ShapeSpec, batched_nms

import matplotlib.pyplot as plt 


def statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir, name):
    fg_proposals = PersistentProposalList(fg_proposals_path)
    bg_proposals = PersistentProposalList(bg_proposals_path)
    print(fg_proposals_path)
    print(os.path.exists(fg_proposals_path))
    fg_proposals.load()
    bg_proposals.load()
    proposals = fg_proposals + bg_proposals

    class_iou_dict = {}
    class_gtclass_dict = {}
    iou_count_dict = {}
    catch_ious = []
    # print(proposals)
    for proposal in proposals:
        pred_boxes, gt_fg_classes, gt_classes, gt_ious, gt_boxes = proposal.pred_boxes, proposal.gt_fg_classes, proposal.gt_classes, proposal.gt_ious, proposal.gt_boxes
        all_gt_classes, all_gt_boxes = proposal.all_gt_classes, proposal.all_gt_boxes
        # print(proposal)
        assert len(pred_boxes) == len(gt_fg_classes) == len(gt_classes) == len(gt_ious) == len(gt_boxes), 'pred lenght not equal'
        for iou, fg_cls, cls in zip(gt_ious, gt_fg_classes, gt_classes):
            iou = round(iou, 1)
            if str(iou) not in iou_count_dict.keys():
                iou_count_dict[str(iou)] = 0
            iou_count_dict[str(iou)] += 1

            if str(cls) not in class_iou_dict.keys():
                class_iou_dict[str(cls)] = []
            class_iou_dict[str(cls)].append(iou)

            if str(cls) not in class_gtclass_dict.keys():
                class_gtclass_dict[str(cls)] = []
            class_gtclass_dict[str(cls)].append(str(fg_cls))

        all_gt_boxes, pred_boxes = Boxes(all_gt_boxes), Boxes(pred_boxes)
        pred_ious = pairwise_iou(all_gt_boxes, pred_boxes)
        catch_ious.append(pred_ious)
    
    x, y = [], []
    
    for iou in range(0, 100, 10):
        iou = iou / 100
        count = 0
        all_gt_count = 0
        for one_catch_ious in catch_ious:
            all_gt_count += one_catch_ious.shape[0]
            one_count = one_catch_ious > iou
            one_count = one_count.any(1).sum()
            count += one_count
        x.append(iou)
        y.append(count / all_gt_count)
        # y.append(count)
    plt.figure(figsize=(15,5))
    plt.bar(x, y, color = '#9999ff', width = 0.05)
    plt.title('all ious count')
    plt.xlabel('iou')
    plt.ylabel('count')
    save_path = os.path.join(save_dir, 'gt_' + name + '.jpg')
    plt.savefig(save_path)
    plt.close()
    

    
    x = sorted(iou_count_dict.keys())
    y = []
    for i in x:
        y.append(iou_count_dict[i])

    plt.figure(figsize=(10,5))
    plt.bar(x, y, color = '#9999ff', width = 0.5) 
    plt.title('all ious count') 
    plt.xlabel('iou') 
    plt.ylabel('count') 
    save_path = os.path.join(save_dir, 'prop_' + name + '.jpg')
    plt.savefig(save_path) 
    plt.close()

def generate_proposals(model, num_classes, dataset_names, cache_root, cfg):
    """Generate foreground proposals and background proposals from `model` and save them to the disk"""
    fg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_fg.json".format(dataset_names[0])))
    bg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_bg.json".format(dataset_names[0])))
    if not (fg_proposals_list.load() and bg_proposals_list.load()):
        for dataset_name in dataset_names:
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=ProposalMapper(cfg, False))
            generator = ProposalGenerator(num_classes=num_classes)
            fg_proposals_list_data, bg_proposals_list_data = inference_on_dataset(model, data_loader, generator)
            fg_proposals_list.extend(fg_proposals_list_data)
            bg_proposals_list.extend(bg_proposals_list_data)
        fg_proposals_list.flush()
        bg_proposals_list.flush()
    return fg_proposals_list, bg_proposals_list

# if __name__ == "__main__":
#     log_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_4/logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase1/cache'
#     # work_root = os.path.join('/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/', log_root, 'logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_0/phase1/cache')
#     figures_dir = os.path.join(log_root, 'figures')
#     if not os.path.exists(figures_dir):
#         os.mkdir(figures_dir)
#     # proposal
#     fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_fg.json')
#     bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_bg.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_source') 

#     fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_fg.json')
#     bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_bg.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_target') 

#     fg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_target') 

#     fg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_bbox_target') 
    


# if __name__ == "__main__":
#     log_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_4/logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_0/phase1/cache'
#     # work_root = os.path.join('/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/', log_root, 'logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_0/phase1/cache')
#     figures_dir = os.path.join(log_root, 'figures')
#     if not os.path.exists(figures_dir):
#         os.mkdir(figures_dir)
#     # proposal
#     fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_fg.json')
#     bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_bg.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_source') 

#     fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_fg.json')
#     bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_bg.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_target') 

#     fg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_target') 

#     fg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_bbox_target') 
    


if __name__ == "__main__":
    log_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_5/logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase1/cache'
    # work_root = os.path.join('/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/', log_root, 'logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_0/phase1/cache')
    figures_dir = os.path.join(log_root, 'figures')
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    # proposal
    fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_fg.json')
    bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_bg.json')
    statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_source') 

    fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_fg.json')
    bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_bg.json')
    statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_target') 

    fg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
    bg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
    statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_target') 

    fg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
    bg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
    statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_bbox_target') 
    


# if __name__ == "__main__":
#     log_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_6/logs/faster_rcnn_R_101_C4/cityscapes2foggy_cascade_test/phase1/cache/'
#     # work_root = os.path.join('/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/', log_root, 'logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_0/phase1/cache')
#     figures_dir = os.path.join(log_root, 'figures')
#     if not os.path.exists(figures_dir):
#         os.mkdir(figures_dir)
#     # proposal
#     fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_fg.json')
#     bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_cityscapes_in_voc_trainval_bg.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_source') 

#     fg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_fg.json')
#     bg_proposals_path = os.path.join(log_root, 'proposal/.._datasets_foggy_cityscapes_in_voc__trainval_bg.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='proposal_target') 

#     fg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_target0') 

#     fg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_fg_1.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_bg_1.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_target1') 

#     fg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_fg_2.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback/.._datasets_foggy_cityscapes_in_voc__trainval_bg_2.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_target2') 

#     fg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_fg_0.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_bg_0.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_bbox_target0')

#     fg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_fg_1.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_bg_1.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_bbox_target1')

#     fg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_fg_2.json')
#     bg_proposals_path = os.path.join(log_root, 'feedback_bbox/.._datasets_foggy_cityscapes_in_voc__trainval_bg_2.json')
#     statistic_proposal(fg_proposals_path, bg_proposals_path, save_dir=figures_dir, name='feedback_bbox_target2') 
    