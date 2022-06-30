"""
`D-adapt: Decoupled Adaptation for Cross-Domain Object Detection <https://openreview.net/pdf?id=VNqaB1g9393>`_.
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from genericpath import exists
import logging
import os
import argparse
from pickle import NONE
from random import random, shuffle
import shutil
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

import matplotlib.pyplot as plt 

def statistic_proposal(proposals, save_path):
    class_iou_dict = {}
    class_gtclass_dict = {}
    iou_count_dict = {}
    catch_ious = []
    for proposal in proposals:
        pred_boxes, gt_fg_classes, gt_classes, gt_ious, gt_boxes = proposal.pred_boxes, proposal.gt_fg_classes, proposal.gt_classes, proposal.gt_ious, proposal.gt_boxes
        all_gt_classes, all_gt_boxes = proposal.all_gt_classes, proposal.all_gt_boxes
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
    
    # catch_ious = np.concatenate(catch_ious, axis=0)
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
    plt.savefig(save_path)
    plt.close()
    

    
    # x = sorted(iou_count_dict.keys())
    # y = []
    # for i in x:
    #     y.append(iou_count_dict[i])

    # plt.figure(figsize=(10,5))
    # plt.bar(x, y, color = '#9999ff', width = 0.5) 
    # plt.title('all ious count') 
    # plt.xlabel('iou') 
    # plt.ylabel('count') 
    # plt.savefig(save_path) 
    # plt.close()



def analyze_proposal(proposal_list, class_names, show_save_dir, crop_save_dir, show_scale=0.5, show_flag=False, crop_flag=False):
    # if show_flag:
    #     if os.path.exists(show_save_dir):
    #         shutil.rmtree(show_save_dir)
    #     os.makedirs(show_save_dir, exist_ok=True)
    # if crop_flag:
    #     if os.path.exists(crop_save_dir):
    #         shutil.rmtree(crop_save_dir)
    #     os.makedirs(crop_save_dir, exist_ok=True)

    if os.path.exists(show_save_dir):
        show_flag = False
    if os.path.exists(crop_save_dir):
        crop_flag = False
    if not show_flag and not crop_flag:
        return

    os.makedirs(show_save_dir, exist_ok=True)
    os.makedirs(crop_save_dir, exist_ok=True)

    # only_crop = True

    # if only_crop:
    #     for proposals in tqdm.tqdm(proposal_list):
    #         file_path = proposals.filename
    #         img = default_loader(file_path)
    #         for idx in range(len(proposals)):
    #             proposal = proposals[idx]
    #             x1, y1, x2, y2 = proposal.pred_boxes
    #             fb_set = proposal.fb_set
    #             top, left, height, width = int(y1), int(x1), int(y2 - y1), int(x2 - x1)
    #             # if self.crop_func is not None:
    #             #     top, left, height, width = self.crop_func(img, top, left, height, width)
    #             img = crop(img, top, left, height, width)
    #             pred_id = proposal.pred_ids
    #             crop_name = os.path.basename(file_path).split('.')[0] + '_{}_proposal_{}.jpg'.format(fb_set, pred_id)
    #             try:
    #                 img.save(os.path.join(crop_save_dir, crop_name))
    #             except:
    #                 print(crop_name)
    #     return




    palette = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
    class_names = class_names + ['bg']
    img_height, img_width = 0, 0
    show_count = 0
    for proposals in tqdm.tqdm(proposal_list):
        file_path = proposals.filename
        img_np_show = cv2.imread(file_path)
        img_np_clean = img_np_show.copy()
        if show_flag and show_count < 20:
            gt_classes, gt_bboxes = proposals.all_gt_classes, proposals.all_gt_boxes
            for idx in range(len(gt_classes)):
                class_id, bbox = gt_classes[idx], gt_bboxes[idx]
                bbox = [int(i) for i in bbox]
                class_name = class_names[class_id]
                cv2.rectangle(img_np_show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), palette['blue'], 2) 
                cv2.putText(img_np_show, '{}'.format(class_name), (bbox[2] + 2, bbox[1] + 2), cv2.FONT_HERSHEY_COMPLEX,
                                0.7, palette['blue'], 1)
            

        for idx in range(len(proposals.pred_classes)):
            pred_box = proposals.pred_boxes[idx]
            pred_class = proposals.pred_classes[idx]
            pred_score = proposals.pred_scores[idx]
            pred_id = int(proposals.pred_ids[idx])
            pred_class_name = class_names[pred_class]
            pred_box = [int(i) for i in pred_box]   # x1 y1 x2 y2
            fb_set = proposals.fb_set
            
            if crop_flag:
                try:
                    crop_img = img_np_clean[pred_box[1]: pred_box[3], pred_box[0]: pred_box[2]]
                    crop_name = os.path.basename(file_path).split('.')[0] + '_{}_proposal_{}.jpg'.format(fb_set, pred_id)
                    # print(pred_box, img_np_clean.shape, crop_name)
                    cv2.imwrite(os.path.join(crop_save_dir, crop_name), crop_img)
                except:
                    print(pred_box, img_np_clean.shape, crop_name)
                    pass
            if show_flag and show_count < 20:
                cv2.rectangle(img_np_show, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), palette['red'], 1) 
                cv2.putText(img_np_show, '{}_{:.2f}'.format(pred_class_name, pred_score), (pred_box[2] + 2, pred_box[1] + 2), cv2.FONT_HERSHEY_COMPLEX,
                                0.7, palette['red'], 1)

        if show_flag and show_count < 20:
            img_height, img_width = img_np_show.shape[:2]
            img_np_show = cv2.resize(img_np_show, (int(img_width * show_scale), int(img_height * show_scale)))
            cv2.imwrite(os.path.join(show_save_dir, os.path.basename(file_path)), img_np_show)

        
        show_count += 1

        if show_count > 20:
            break


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

def generate_proposals_only_inference(model, num_classes, dataset_names, cache_root, cfg):
    """Generate foreground proposals and background proposals from `model` and save them to the disk"""
    if True:
    # if not (fg_proposals_list.load() and bg_proposals_list.load()):
        for dataset_name in dataset_names:
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=ProposalMapper(cfg, False))
            generator = ProposalGenerator(num_classes=num_classes)
            fg_proposals_list_data, bg_proposals_list_data = inference_on_dataset(model, data_loader, generator)
            # fg_proposals_list.extend(fg_proposals_list_data)
            # bg_proposals_list.extend(bg_proposals_list_data)
        # fg_proposals_list.flush()
        # bg_proposals_list.flush()
    print('using pre prop')
    fg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_fg.json".format(dataset_names[0])))
    bg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_bg.json".format(dataset_names[0])))
    fg_proposals_list.load()
    bg_proposals_list.load()
    return fg_proposals_list, bg_proposals_list


def generate_category_labels(prop, category_adaptor, cache_filename, crop_img_dir=None, cascade_id=0, update_score=False):
    """Generate category labels for each proposals in `prop` and save them to the disk"""
    if update_score:
        print('update category scores ......')
    prop_w_category = PersistentProposalList(cache_filename)
    # if cascade_id == 0:
    #     prop_w_category.load()
    #     return prop_w_category
    # if True:
    
    if not prop_w_category.load():
        for p in prop:
            prop_w_category.append(p)

        data_loader_test = category_adaptor.prepare_test_data(flatten(prop_w_category), crop_img_dir=crop_img_dir)
        predictions, scores = category_adaptor.predict(data_loader_test)
        for p in prop_w_category:
            p.pred_classes = np.array([predictions.popleft() for _ in range(len(p))])

            p.pred_classes = copy.deepcopy(p.gt_classes)

            if update_score:
                p.pred_scores = np.array([scores.popleft() for _ in range(len(p))])
        prop_w_category.flush()
    return prop_w_category


def generate_bounding_box_labels_0(prop, bbox_adaptor, class_names, cache_filename, crop_img_dir=None, remove_bg=False):
    """Generate bounding box labels for each proposals in `prop` and save them to the disk"""
    prop_w_bbox = PersistentProposalList(cache_filename)
    # if not prop_w_bbox.load():
    if True:
        # remove (predicted) background proposals
        for p in prop:
            keep_indices = (0 <= p.pred_classes) & (p.pred_classes < len(class_names))
            prop_w_bbox.append(p[keep_indices])
        data_loader_test = bbox_adaptor.prepare_test_data(flatten(prop_w_bbox), crop_img_dir=crop_img_dir)
        predictions = bbox_adaptor.predict(data_loader_test)
        for p in prop_w_bbox:
            p.pred_boxes = np.array([predictions.popleft() for _ in range(len(p))])
        if not remove_bg:
            for p in prop:
            # for i in range(len(prop)):
                keep_indices = (0 <= p.pred_classes) & (p.pred_classes < len(class_names))
                prop_w_bbox.append(p[~keep_indices])
        prop_w_bbox.flush()
    return prop_w_bbox

def remove_prop_bg(prop, cache_filename, num_classes):
    prop_w_bbox = PersistentProposalList(cache_filename)
    for p in prop:
        keep_indices = (0 <= p.pred_classes) & (p.pred_classes < num_classes)
        prop_w_bbox.append(p[keep_indices])
    prop_w_bbox.flush()
    return prop_w_bbox



def generate_bounding_box_labels(prop, bbox_adaptor, class_names, cache_filename, crop_img_dir=None, remove_bg=False, cascade_id=0):
    """Generate bounding box labels for each proposals in `prop` and save them to the disk"""
    prop_w_bbox = PersistentProposalList(cache_filename)
    prop_w_bbox_fg = PersistentProposalList()
    prop_w_bbox_bg = PersistentProposalList()
    
    # if cascade_id == 0:
    #     prop_w_bbox.load()
    #     return prop_w_bbox
    if not prop_w_bbox.load():
    # if True:
        # remove (predicted) background proposals
        indices_fg_ls = []
        indices_bg_ls = []
        for p in prop:
            keep_indices = (0 <= p.pred_classes) & (p.pred_classes < len(class_names))
            # prop_w_bbox_fg.append(copy.deepcopy(p[keep_indices]))
            # prop_w_bbox_bg.append(copy.deepcopy(p[~keep_indices]))
            prop_w_bbox_fg.append(p[keep_indices])
            prop_w_bbox_bg.append(p[~keep_indices])
            indices_fg_ls.append(np.where(keep_indices)[0].tolist())
            indices_bg_ls.append(np.where(~keep_indices)[0].tolist())
            # lenght = len(p) // 2
            # prop_w_bbox_fg.append(copy.deepcopy(p[:lenght]))
            # prop_w_bbox_bg.append(copy.deepcopy(p[lenght:]))
        data_loader_test = bbox_adaptor.prepare_test_data(flatten(prop_w_bbox_fg), crop_img_dir=crop_img_dir)
        predictions = bbox_adaptor.predict(data_loader_test)
        for p in prop_w_bbox_fg:
            p.pred_boxes = np.array([predictions.popleft() for _ in range(len(p))])

            p.pred_boxes = copy.deepcopy(p.gt_boxes)

        print('generate_bounding_box_labels remove_bg: {}'.format(remove_bg))
        if not remove_bg:
            print('Do not remove bg ......')
            for i in range(len(prop)):
                p_fg = prop_w_bbox_fg[i]
                p_bg = prop_w_bbox_bg[i]
                p_fg.extend(p_bg)
                # p = prop[i]
                indices_fg = indices_fg_ls[i]
                indices_bg = indices_bg_ls[i]
                indices = indices_fg + indices_bg
                new_indices = []
                # print(indices)
                for j in range(len(indices)):
                    new_indices.append(indices.index(j))
                # print(new_indices)
                p_fg = p_fg[new_indices]
                prop_w_bbox.append(p_fg)
        else:
            print('Remove bg ......')
            for i in range(len(prop)):
                p_fg = prop_w_bbox_fg[i]
                prop_w_bbox.append(p_fg)
        prop_w_bbox.flush()
    return prop_w_bbox

def compare_proposals(proposal_list_a, proposal_list_b):
    # print('comparing proposals')
    for proposal_a, proposal_b in zip(proposal_list_a, proposal_list_b):
        pred_boxes_a, gt_fg_classes_a, gt_classes_a, gt_ious_a, gt_boxes_a = proposal_a.pred_boxes, proposal_a.gt_fg_classes, proposal_a.gt_classes, proposal_a.gt_ious, proposal_a.gt_boxes
        pred_boxes_b, gt_fg_classes_b, gt_classes_b, gt_ious_b, gt_boxes_b = proposal_b.pred_boxes, proposal_b.gt_fg_classes, proposal_b.gt_classes, proposal_b.gt_ious, proposal_b.gt_boxes

        pred_boxes_diff = np.array((pred_boxes_a - pred_boxes_b)).sum()
        gt_fg_classes_diff = np.array((gt_fg_classes_a - gt_fg_classes_b)).sum()
        gt_classes_diff = np.array((gt_classes_a - gt_classes_b)).sum()
        gt_ious_diff = np.array((gt_ious_a - gt_ious_b)).sum()
        gt_boxes_diff = np.array((gt_boxes_a - gt_boxes_b)).sum()
        # print(pred_boxes_diff, gt_fg_classes_diff, gt_classes_diff, gt_ious_diff, gt_boxes_diff)
        if pred_boxes_diff != 0:
            print('Diff in pred_boxes --------')
            print(pred_boxes_a, pred_boxes_b)
        if gt_fg_classes_diff != 0:
            print('Diff in gt_fg_classes_diff --------')
            print(gt_fg_classes_a, gt_fg_classes_b)
        if gt_classes_diff != 0:
            print('Diff in gt_classes_diff --------')
            print(gt_classes_a, gt_classes_b)
        if gt_ious_diff != 0:
            print('Diff in gt_ious_diff --------')
            print(gt_ious_a, gt_ious_b)
        if gt_boxes_diff != 0:
            print('Diff in gt_boxes_diff --------')
            print(gt_boxes_a, gt_boxes_b)
    print('finish comparing!')

def arrays_equal(proposal_list_a, proposal_list_b):
    for proposal_a, proposal_b in zip(proposal_list_a, proposal_list_b):
        for k in proposal_a.keys():
            a = eval('proposal_a.{}'.format(k))
            b = eval('proposal_b.{}'.format(k))
            # if a.shape != b.shape:
            #     print(a)
            #     print(b)
            #     return False
            if type(a) is not np.ndarray:
                if a != b:
                    print(a)
                    print(b)
                    return False
            else:
                for ai, bi in zip(a.flat, b.flat):
                    if ai != bi:
                        print(a)
                        print(b)
                        return False
    return True

def train(model, logger, cfg, args, args_cls, args_box):
    model.train()
    distributed = comm.get_world_size() > 1
    if distributed:
        model_without_parallel = model.module
    else:
        model_without_parallel = model

    # define optimizer and lr scheduler
    params = []
    for module, lr in model_without_parallel.get_parameters(cfg.SOLVER.BASE_LR):
        params.extend(
            get_default_optimizer_params(
                module,
                base_lr=lr,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )
        )
    optimizer = maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    scheduler = utils.build_lr_scheduler(cfg, optimizer)

    # resume from the last checkpoint
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    args.sources = utils.build_dataset(args.sources[::2], args.sources[1::2])
    args.targets = utils.build_dataset(args.targets[::2], args.targets[1::2])
    args.test = utils.build_dataset(args.test[::2], args.test[1::2])

    # generate proposals from detector
    classes = MetadataCatalog.get(args.targets[0]).thing_classes
    cache_proposal_root = os.path.join(cfg.OUTPUT_DIR, "cache", "proposal")

    # pre_cache_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_3/logs/faster_rcnn_R_101_C4/cityscapes2foggy_ori_cache/phase1/cache'
    # pre_cache_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_4/logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase1/cache'
    # pre_cache_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_4/logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_0/phase1/cache'
    pre_cache_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_6/logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_new_3/phase1/cache'

    if args.use_pre_cache:
    # if True:
        if not os.path.exists(cache_proposal_root):
            print('using pre cahce ......')
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, "cache"), exist_ok=True)
            shutil.copytree(os.path.join(pre_cache_root, 'proposal'), cache_proposal_root, dirs_exist_ok=True)
            # os.symlink(os.path.join(pre_cache_root, 'proposal'), cache_proposal_root)
        # print('using pre prop beging ..... ')
        # prop_t_fg, prop_t_bg = generate_proposals_only_inference(model, len(classes), args.targets, cache_proposal_root, cfg)
        # prop_s_fg, prop_s_bg = generate_proposals_only_inference(model, len(classes), args.sources, cache_proposal_root, cfg)    
        prop_t_fg, prop_t_bg = generate_proposals(model, len(classes), args.targets, cache_proposal_root, cfg)
        prop_s_fg, prop_s_bg = generate_proposals(model, len(classes), args.sources, cache_proposal_root, cfg)
    else:
        prop_t_fg, prop_t_bg = generate_proposals(model, len(classes), args.targets, cache_proposal_root, cfg)
        prop_s_fg, prop_s_bg = generate_proposals(model, len(classes), args.sources, cache_proposal_root, cfg)



    print('\norigin map:')
    map_evaluater_target = PascalVOCDetectionPerClassEvaluatorMAP(args.targets[0])
    # map_evaluater_target.reset()
    # map_evaluater_target.process(prop_t_fg + prop_t_bg, len(classes))
    # results = map_evaluater_target.evaluate()
    # print(results)

    map_evaluater_source = PascalVOCDetectionPerClassEvaluatorMAP(args.sources[0])
    # map_evaluater_source.reset()
    # map_evaluater_source.process(prop_s_fg + prop_s_bg, len(classes))
    # results = map_evaluater_source.evaluate()
    # print(results)

    model = model.to(torch.device('cpu'))


    # show_flag = False
    show_flag = True
    crop_flag = True
    gt_pred_show_root = os.path.join(cfg.OUTPUT_DIR, "cache", "show_gt_pred")
    crop_proposal_save_root = os.path.join(cfg.OUTPUT_DIR, "cache", "crop_propals")
    source_crop_proposal_save_root = os.path.join(crop_proposal_save_root, 'source')

    if args.use_pre_crop:
        if not os.path.exists(source_crop_proposal_save_root):
            os.makedirs(crop_proposal_save_root, exist_ok=True)
            os.symlink(os.path.join(pre_cache_root, 'crop_propals', 'source'), 
                        source_crop_proposal_save_root)
    analyze_proposal(prop_s_fg + prop_s_bg, classes, show_save_dir=os.path.join(gt_pred_show_root, 'source'), 
                    crop_save_dir=source_crop_proposal_save_root, show_scale=0.6, show_flag=show_flag, crop_flag=crop_flag) 

    target_crop_proposal_save_root = os.path.join(crop_proposal_save_root, 'target_0')
    if args.use_pre_crop:
        if not os.path.exists(target_crop_proposal_save_root):
            os.symlink(os.path.join(pre_cache_root, 'crop_propals', 'target_0'),
                        target_crop_proposal_save_root)
    analyze_proposal(prop_t_fg + prop_t_bg, classes, show_save_dir=os.path.join(gt_pred_show_root, 'target_0'),
                    crop_save_dir=target_crop_proposal_save_root, show_scale=0.6, show_flag=show_flag, crop_flag=crop_flag)
    
    # statistic_proposal(prop_s_fg + prop_s_bg, save_path=os.path.join(cfg.OUTPUT_DIR, 'statistic_proposal_source.jpg'))
    # statistic_proposal(prop_t_fg + prop_t_bg, save_path=os.path.join(cfg.OUTPUT_DIR, 'statistic_proposal_target.jpg'))
    # exit()

    prop_t_fg_category = None
    prop_t_bg_category = None

    # cascade_flag_category = None
    # cascade_flag_bbox = None
    # cascade_flag_category = [True, True, True]
    # cascade_flag_bbox = [False, False, False]

    # if cascade_flag_category is None:
    #     cascade_flag_category = [True for i in range(args.num_cascade)]
    # if cascade_flag_bbox is None:
    #     cascade_flag_bbox = [True for i in range(args.num_cascade)]

    cascade_flag_category = [True if args.cascade_flag_category[i] == '1' else False for i in range(len(args.cascade_flag_category))]
    cascade_flag_bbox = [True if args.cascade_flag_bbox[i] == '1' else False for i in range(len(args.cascade_flag_bbox))]
    ignored_scores_ls = eval(args.ignored_scores_ls)
    ignored_ious_ls = eval(args.ignored_ious_ls)
    assert args.num_cascade == len(cascade_flag_category), 'num_cascade {} not equals to cascade_flag_category {}'.format(args.num_cascade, len(cascade_flag_category))
    assert args.num_cascade == len(cascade_flag_bbox), 'num_cascade {} not equals to cascade_flag_bbox {}'.format(args.num_cascade, len(cascade_flag_bbox))
    assert args.num_cascade == len(ignored_scores_ls), 'num_cascade {} not equals to ignored_scores_ls {}'.format(args.num_cascade, len(ignored_scores_ls))
    assert args.num_cascade == len(ignored_ious_ls), 'num_cascade {} not equals to ignored_ious_ls {}'.format(args.num_cascade, len(ignored_ious_ls))
    
    update_feedback_root = os.path.join(cfg.OUTPUT_DIR, "cache", "update")
    # # update gt pred bbox
    # prop_s_fg = update_proposal(prop_s_fg, len(classes), ignored_ious_ls[0])
    # prop_s_bg = update_proposal(prop_s_bg, len(classes), ignored_ious_ls[0])
    # prop_t_fg = update_proposal(prop_t_fg, os.path.join(update_feedback_root, "{}_category_fg_zero.json".format(args.targets[0])), len(classes), ignored_ious_ls[0])
    # prop_t_bg = update_proposal(prop_t_bg, os.path.join(update_feedback_root, "{}_category_fg_zero.json".format(args.targets[0])), len(classes), ignored_ious_ls[0])
    # analyze_proposal(prop_t_fg + prop_t_bg, classes, show_save_dir=os.path.join(gt_pred_show_root, 'target_update1'),
    #     crop_save_dir=target_crop_proposal_save_root, show_scale=0.6, show_flag=show_flag, crop_flag=crop_flag)

    
    # map_evaluater_target.reset()
    # map_evaluater_target.process(prop_t_fg + prop_t_bg, len(classes))
    # results = map_evaluater_target.evaluate()
    # print('updated target proposal mAP:')
    # print(results)

    # '''
    for cascade_id in range(args.num_cascade):
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>> Cascade Phase {} >>>>>>>>>>>'.format(cascade_id))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

        if args.update_cache:
            if cascade_id != 0:
                crop_proposal_save_root = os.path.join(cfg.OUTPUT_DIR, "cache", "crop_propals_update")
                if not os.path.exists(crop_proposal_save_root):
                    os.makedirs(crop_proposal_save_root)
                target_crop_proposal_save_root = os.path.join(crop_proposal_save_root, 'target_{}'.format(cascade_id))
                show_save_dir = os.path.join(gt_pred_show_root, 'target_{}'.format(cascade_id))
                analyze_proposal(prop_t_fg + prop_t_bg, classes, show_save_dir=show_save_dir, 
                                crop_save_dir=target_crop_proposal_save_root, show_scale=0.6, show_flag=show_flag, crop_flag=crop_flag)

        if cascade_flag_category[cascade_id]:
            # train the category adaptor
            category_adaptor = category_adaptation.CategoryAdaptor(classes, os.path.join(cfg.OUTPUT_DIR, "cls_{}".format(cascade_id)), args_cls)
            if args.update_proposal:
            # and cascade_id == 0:
            # if True:
                print('update proposal for category at cascade_{}: ignore_scores {}, ignore_ious {}'.format(cascade_id, ignored_scores_ls[cascade_id], ignored_ious_ls[cascade_id]))
                # prop_s_fg = update_proposal(prop_s_fg, os.path.join(update_feedback_root, "{}_category_fg_{}.json".format(args.sources[0], cascade_id)), len(classes), ignored_ious_ls[cascade_id])
                # prop_s_bg = update_proposal(prop_s_bg, os.path.join(update_feedback_root, "{}_category_bg_{}.json".format(args.sources[0], cascade_id)), len(classes), ignored_ious_ls[cascade_id])
                prop_t_fg = update_proposal(prop_t_fg, os.path.join(update_feedback_root, "{}_category_fg_{}.json".format(args.targets[0], cascade_id)), len(classes), ignored_ious_ls[cascade_id])
                prop_t_bg = update_proposal(prop_t_bg, os.path.join(update_feedback_root, "{}_category_bg_{}.json".format(args.targets[0], cascade_id)), len(classes), ignored_ious_ls[cascade_id])
                # analyze_proposal(prop_t_fg + prop_t_bg, classes, show_save_dir=os.path.join(gt_pred_show_root, 'target_update'),
                #     crop_save_dir=target_crop_proposal_save_root, show_scale=0.6, show_flag=show_flag, crop_flag=crop_flag)
                
                
                
                # 逐条复制生成新list，看是否管用
                # 作统计图时，查看每个类别剩余的prop数量

                # prop_s_fg_update = update_proposal(prop_s_fg, len(classes), ignored_ious_ls[cascade_id])
                # prop_s_bg_update = update_proposal(prop_s_bg, len(classes), ignored_ious_ls[cascade_id])
                # prop_t_fg_update = update_proposal(prop_t_fg, len(classes), ignored_ious_ls[cascade_id])
                # prop_t_bg_update = update_proposal(prop_t_bg, len(classes), ignored_ious_ls[cascade_id])

                # print(arrays_equal(prop_s_fg, prop_s_fg_update))
                # print(arrays_equal(prop_s_bg, prop_s_bg_update))
                # print(arrays_equal(prop_t_fg, prop_t_fg_update))
                # print(arrays_equal(prop_t_bg, prop_t_bg_update))

                # compare_proposals(prop_s_fg, prop_s_fg_update)
                # compare_proposals(prop_s_bg, prop_s_bg_update)
                # compare_proposals(prop_t_fg, prop_t_fg_update)
                # compare_proposals(prop_t_bg, prop_t_bg_update)

                map_evaluater_target.reset()
                map_evaluater_target.process(prop_t_fg + prop_t_bg, len(classes))
                results = map_evaluater_target.evaluate()
                print('updated target proposal mAP:')
                print(results)

                map_evaluater_source.reset()
                map_evaluater_source.process(prop_s_fg + prop_s_bg, len(classes))
                results = map_evaluater_source.evaluate()
                print('updated source proposal mAP:')
                print(results)

                # exit()

            
            # if cascade_id == 0:
            #     category_adaptor.load_checkpoint()
            #     data_loader_validation = category_adaptor.prepare_validation_data(prop_t_fg + prop_t_bg, 
            #                                                                     crop_img_dir=target_crop_proposal_save_root)
            #     category_adaptor.validate(data_loader_validation, category_adaptor.model, category_adaptor.class_names, category_adaptor.args)
            # else:

            if not category_adaptor.load_checkpoint():
            # if True:
                data_loader_source = category_adaptor.prepare_training_data(prop_s_fg + prop_s_bg, True, crop_img_dir=source_crop_proposal_save_root, 
                                                                            ignored_scores=ignored_scores_ls[cascade_id], ignored_ious=ignored_ious_ls[cascade_id])
                data_loader_target = category_adaptor.prepare_training_data(prop_t_fg + prop_t_bg, False, crop_img_dir=target_crop_proposal_save_root,
                                                                            ignored_scores=ignored_scores_ls[cascade_id], ignored_ious=ignored_ious_ls[cascade_id])
                data_loader_validation = category_adaptor.prepare_validation_data(prop_t_fg + prop_t_bg, 
                                                                                crop_img_dir=target_crop_proposal_save_root)
                category_adaptor.fit(data_loader_source, data_loader_target, data_loader_validation)

            # generate category labels for each proposals
            cache_feedback_root = os.path.join(cfg.OUTPUT_DIR, "cache", "feedback")
            prop_t_fg = generate_category_labels(
                prop_t_fg, category_adaptor, os.path.join(cache_feedback_root, "{}_fg_{}.json".format(args.targets[0], cascade_id)), 
                crop_img_dir=target_crop_proposal_save_root, cascade_id=cascade_id, update_score=args.update_score
            )
            prop_t_bg = generate_category_labels(
                prop_t_bg, category_adaptor, os.path.join(cache_feedback_root, "{}_bg_{}.json".format(args.targets[0], cascade_id)), 
                crop_img_dir=target_crop_proposal_save_root, cascade_id=cascade_id, update_score=args.update_score
            )
            prop_t_fg_category = copy.deepcopy(prop_t_fg)
            prop_t_bg_category = copy.deepcopy(prop_t_bg)
            category_adaptor.model.to(torch.device("cpu"))

            map_evaluater_target.reset()
            map_evaluater_target.process(prop_t_fg + prop_t_bg, num_classes=len(classes))
            results = map_evaluater_target.evaluate()
            print('generated target proposal mAP:')
            print(results)

            # map_evaluater_source.reset()
            # map_evaluater_source.process(prop_s_fg + prop_s_bg)
            # results = map_evaluater_source.evaluate()
            # print('generated source proposal mAP:')
            # print(results)


        if args.bbox_refine and cascade_flag_bbox[cascade_id]:
            # train the bbox adaptor
            bbox_adaptor = bbox_adaptation.BoundingBoxAdaptor(classes, os.path.join(cfg.OUTPUT_DIR, "bbox_{}".format(cascade_id)), args_box)
            # if args.update_proposal:
            if False:
                print('update proposal for bbox at cascade_{}: ignore_scores {}, ignore_ious {}'.format(cascade_id, ignored_scores_ls[cascade_id], ignored_ious_ls[cascade_id]))
                prop_s_fg = update_proposal(prop_s_fg, len(classes), ignored_ious_ls[cascade_id])
                prop_s_bg = update_proposal(prop_s_bg, len(classes), ignored_ious_ls[cascade_id])
                prop_t_fg = update_proposal(prop_t_fg, len(classes), ignored_ious_ls[cascade_id])
                prop_t_bg = update_proposal(prop_t_bg, len(classes), ignored_ious_ls[cascade_id])

                map_evaluater_target.reset()
                map_evaluater_target.process(prop_t_fg + prop_t_bg, len(classes))
                results = map_evaluater_target.evaluate()
                print('updated target proposal mAP:')
                print(results)

                map_evaluater_source.reset()
                map_evaluater_source.process(prop_s_fg + prop_s_bg, len(classes))
                results = map_evaluater_source.evaluate()
                print('updated source proposal mAP:')
                print(results)
            
            if not bbox_adaptor.load_checkpoint():
            # if True:
                data_loader_source = bbox_adaptor.prepare_training_data(prop_s_fg, True, crop_img_dir=source_crop_proposal_save_root)
                data_loader_target = bbox_adaptor.prepare_training_data(prop_t_fg, False, crop_img_dir=target_crop_proposal_save_root)
                data_loader_validation = bbox_adaptor.prepare_validation_data(prop_t_fg, crop_img_dir=target_crop_proposal_save_root)
                # data_loader_validation_source = bbox_adaptor.prepare_validation_data_source(prop_s_fg, crop_img_dir=source_crop_proposal_save_root)
                print('data_loader_validation baseline: ......')
                bbox_adaptor.validate_baseline(data_loader_validation)
                # print('data_loader_validation_source baseline: ......')
                # bbox_adaptor.validate_baseline(data_loader_validation_source)
                bbox_adaptor.fit(data_loader_source, data_loader_target, data_loader_validation)
                # bbox_adaptor.fit(data_loader_source, data_loader_target, data_loader_validation, data_loader_validation_source)

            # generate bounding box labels for each proposals
            cache_feedback_root = os.path.join(cfg.OUTPUT_DIR, "cache", "feedback_bbox")
            
            remove_bg_flag = cascade_id == (args.num_cascade - 1)
            
            # prop_t_fg_refined = generate_bounding_box_labels(
            prop_t_fg = generate_bounding_box_labels(
                prop_t_fg, bbox_adaptor, classes, os.path.join(cache_feedback_root, "{}_fg_{}.json".format(args.targets[0], cascade_id)), 
                crop_img_dir=target_crop_proposal_save_root, remove_bg=remove_bg_flag, cascade_id=cascade_id
            )
            ### prop_t_bg_refined = generate_bounding_box_labels(
            prop_t_bg = generate_bounding_box_labels(
                prop_t_bg, bbox_adaptor, classes, os.path.join(cache_feedback_root, "{}_bg_{}.json".format(args.targets[0], cascade_id)), 
                crop_img_dir=target_crop_proposal_save_root, remove_bg=remove_bg_flag, cascade_id=cascade_id
            )
            bbox_adaptor.model.to(torch.device("cpu"))

            map_evaluater_target.reset()
            map_evaluater_target.process(prop_t_fg + prop_t_bg, len(classes))
            results = map_evaluater_target.evaluate()
            print('generated target proposal mAP:')
            print(results)

            # map_evaluater_source.reset()
            # map_evaluater_source.process(prop_s_fg + prop_s_bg)
            # results = map_evaluater_source.evaluate()
            # print('generated source proposal mAP:')
            # print(results)


    prop_t_fg += prop_t_fg_category
    prop_t_bg += prop_t_bg_category

    # '''
    # prop_t_bg = remove_prop_bg(prop_t_bg, os.path.join(update_feedback_root, "{}_bg_rm.json".format(args.targets[0])), len(classes))
    # prop_t_fg = remove_prop_bg(prop_t_fg, os.path.join(update_feedback_root, "{}_fg_rm.json".format(args.targets[0])), len(classes))
    # analyze_proposal(prop_t_fg + prop_t_bg, classes, show_save_dir=os.path.join(gt_pred_show_root, 'target_update2'),
    #     crop_save_dir=target_crop_proposal_save_root, show_scale=0.6, show_flag=show_flag, crop_flag=crop_flag)


    # remove_bg
    if args.reduce_proposals:
        # remove proposals
        prop_t_bg_new = []
        for p in prop_t_bg:
            keep_indices = p.pred_classes == len(classes)
            prop_t_bg_new.append(p[keep_indices])
        prop_t_bg = prop_t_bg_new

        prop_t_fg_new = []
        for p in prop_t_fg:
            prop_t_fg_new.append(p[:20])
        prop_t_fg = prop_t_fg_new

    model = model.to(torch.device(cfg.MODEL.DEVICE))
    # Data loading code
    train_source_dataset = get_detection_dataset_dicts(args.sources)
    train_source_loader = build_detection_train_loader(dataset=train_source_dataset, cfg=cfg)
    train_target_dataset = get_detection_dataset_dicts(args.targets, proposals_list=prop_t_fg+prop_t_bg)

    mapper = DatasetMapper(cfg, precomputed_proposal_topk=1000, augmentations=utils.build_augmentation(cfg, True))
    train_target_loader = build_detection_train_loader(dataset=train_target_dataset, cfg=cfg, mapper=mapper,
                                                       total_batch_size=cfg.SOLVER.IMS_PER_BATCH)

    # training the object detector
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data_s, data_t, iteration in zip(train_source_loader, train_target_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            optimizer.zero_grad()

            # compute losses and gradient on source domain
            loss_dict_s = model(data_s)
            losses_s = sum(loss_dict_s.values())
            assert torch.isfinite(losses_s).all(), loss_dict_s

            loss_dict_reduced_s = {"{}_s".format(k): v.item() for k, v in comm.reduce_dict(loss_dict_s).items()}
            losses_reduced_s = sum(loss for loss in loss_dict_reduced_s.values())
            losses_s.backward()

            # compute losses and gradient on target domain
            loss_dict_t = model(data_t, labeled=False)
            losses_t = sum(loss_dict_t.values())
            assert torch.isfinite(losses_t).all()

            loss_dict_reduced_t = {"{}_t".format(k): v.item() for k, v in comm.reduce_dict(loss_dict_t).items()}
            (losses_t * args.trade_off).backward()

            if comm.is_main_process():
                storage.put_scalars(total_loss_s=losses_reduced_s, **loss_dict_reduced_s, **loss_dict_reduced_t)

            # do SGD step
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # evaluate on validation set
            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                utils.validate(model, logger, cfg, args)
                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def main(args, args_cls, args_box):
    logger = logging.getLogger("detectron2")
    cfg = utils.setup(args)

    # create model
    model = models.__dict__[cfg.MODEL.META_ARCHITECTURE](cfg, finetune=args.finetune)
    model.to(torch.device(cfg.MODEL.DEVICE))
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return utils.validate(model, logger, cfg, args)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    train(model, logger, cfg, args, args_cls, args_box)

    # evaluate on validation set
    return utils.validate(model, logger, cfg, args)


if __name__ == "__main__":
    args_cls, argv = category_adaptation.CategoryAdaptor.get_parser().parse_known_args()
    print("Category Adaptation Args:")
    pprint.pprint(args_cls)

    args_box, argv = bbox_adaptation.BoundingBoxAdaptor.get_parser().parse_known_args(args=argv)
    print("Bounding Box Adaptation Args:")
    pprint.pprint(args_box)

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--num-cascade', default=1, type=int,
                        help='num-cascade')
    parser.add_argument('--cascade-flag-category', default='1', type=str,
                        help='num-cascade')
    parser.add_argument('--cascade-flag-bbox', default='1', type=str,
                        help='num-cascade')

    parser.add_argument('--ignored-scores-ls', type=str, default='[[0.05, 0.5]]')
    parser.add_argument('--ignored-ious-ls', type=str, default='[[0.4, 0.5]]')

    parser.add_argument('--use-pre-cache', default=0, type=int,
                        help='pre cache')
    parser.add_argument('--use-pre-crop', default=0, type=int,
                        help='pre crop')
    parser.add_argument('--update-score', default=0, type=int,
                        help='pre cache')
    parser.add_argument('--update-cache', default=0, type=int,
                        help='pre cache')
    parser.add_argument('--update-proposal', default=0, type=int,
                        help='pre cache')

    # dataset parameters
    parser.add_argument('-s', '--sources', nargs='+', help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', help='target domain(s)')
    parser.add_argument('--test', nargs='+', help='test domain(s)')
    # model parameters
    parser.add_argument('--finetune', action='store_true',
                        help='whether use 10x smaller learning rate for backbone')
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
             "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument('--trade-off', default=1., type=float,
                        help='trade-off hyper-parameter for losses on target domain')
    parser.add_argument('--bbox-refine', action='store_true',
                        help='whether perform bounding box refinement')
    parser.add_argument('--reduce-proposals', action='store_true',
                        help='whether remove some low-quality proposals.'
                             'Helpful for RetinaNet')
    # training parameters
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0,
                        help="the rank of this machine (unique per machine)")
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args, argv = parser.parse_known_args(argv)
    print("Detection Args:")
    pprint.pprint(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, args_cls, args_box),
    )
