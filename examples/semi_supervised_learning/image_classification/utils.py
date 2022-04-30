"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import math
import sys
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataset import Subset, ConcatDataset
import torchvision
import torchvision.transforms as T
from torchvision.datasets.folder import default_loader
import timm
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
from tllib.modules.classifier import Classifier
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrained=True, pretrained_checkpoint=None):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrained)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrained)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        backbone.load_state_dict(pretrained_dict, strict=False)
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


def get_dataset(dataset_name, num_samples_per_class, root, labeled_train_transform, val_transform,
                unlabeled_train_transform=None, seed=0):
    if unlabeled_train_transform is None:
        unlabeled_train_transform = labeled_train_transform

    if dataset_name == 'OxfordFlowers102':
        dataset = datasets.__dict__[dataset_name]
        base_dataset = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        # create labeled and unlabeled splits
        labeled_idxes, unlabeled_idxes = x_u_split(num_samples_per_class, base_dataset.num_classes,
                                                   base_dataset.targets, seed=seed)
        # labeled subset
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        labeled_train_dataset.num_classes = base_dataset.num_classes
        # unlabeled subset
        base_dataset = dataset(root=root, split='train', transform=unlabeled_train_transform, download=False)
        unlabeled_train_dataset = ConcatDataset([
            Subset(base_dataset, unlabeled_idxes),
            dataset(root=root, split='validation', download=False, transform=unlabeled_train_transform)
        ])
        val_dataset = dataset(root=root, split='test', download=False, transform=val_transform)
    else:
        dataset = datasets.__dict__[dataset_name]
        base_dataset = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        # create labeled and unlabeled splits
        labeled_idxes, unlabeled_idxes = x_u_split(num_samples_per_class, base_dataset.num_classes,
                                                   base_dataset.targets, seed=seed)
        # labeled subset
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        labeled_train_dataset.num_classes = base_dataset.num_classes
        # unlabeled subset
        base_dataset = dataset(root=root, split='train', transform=unlabeled_train_transform, download=False)
        unlabeled_train_dataset = Subset(base_dataset, unlabeled_idxes)
        val_dataset = dataset(root=root, split='test', download=False, transform=val_transform)
    return labeled_train_dataset, unlabeled_train_dataset, val_dataset


def x_u_split(num_samples_per_class, num_classes, labels, seed):
    """
    Construct labeled and unlabeled subsets, where the labeled subset is class balanced. Note that the resulting
    subsets are **deterministic** with the same random seed.
    """
    labels = np.array(labels)
    assert num_samples_per_class * num_classes <= len(labels)
    random_state = np.random.RandomState(seed)

    # labeled subset
    labeled_idxes = []
    for i in range(num_classes):
        ith_class_idxes = np.where(labels == i)[0]
        ith_class_idxes = random_state.choice(ith_class_idxes, num_samples_per_class, False)
        labeled_idxes.extend(ith_class_idxes)

    # unlabeled subset
    unlabeled_idxes = [i for i in range(len(labels)) if i not in labeled_idxes]
    return labeled_idxes, unlabeled_idxes


def get_train_transform(resizing='default', random_horizontal_flip=True, auto_augment=None,
                        norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.2, 1.))
    elif resizing == 'cifar':
        transform = T.Compose([
            T.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            ResizeImage(224)
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'cifar':
        transform = ResizeImage(224)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions


def get_cosine_scheduler_with_warmup(optimizer, T_max, num_cycles=7. / 16., num_warmup_steps=0,
                                     last_epoch=-1):
    """
    Cosine learning rate scheduler from `FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence (NIPS 2020) <https://arxiv.org/abs/2001.07685>`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        num_cycles (float): A scalar that controls the shape of cosine function. Default: 7/16.
        num_warmup_steps (int): Number of iterations to warm up. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, T_max - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def validate(val_loader, model, args, device, num_classes):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    confmat = ConfusionMatrix(num_classes)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        acc_global, acc_per_class, iu = confmat.compute()
        mean_cls_acc = acc_per_class.mean().item() * 100
        print(' * Mean Cls {:.3f}'.format(mean_cls_acc))

    return top1.avg, mean_cls_acc
