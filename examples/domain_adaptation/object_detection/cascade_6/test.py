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


pre_cache_root = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_4/logs/faster_rcnn_R_101_C4/cityscapes2foggy_update_0/phase1/cache'

keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)