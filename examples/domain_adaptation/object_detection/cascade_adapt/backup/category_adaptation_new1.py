"""
Training a category adaptor
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from gettext import npgettext
import random
import time
import warnings
import sys
import argparse
import os.path as osp
from collections import deque
import tqdm
from typing import List

import prettytable

import torch
from torch import Tensor
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import detectron2.utils.comm as comm

sys.path.append('../../../..')
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from tllib.alignment.d_adapt.proposal import ProposalDataset, flatten, Proposal
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy, ConfusionMatrix, compute_confusionmatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.vision.transforms import ResizeImage

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distributed = comm.get_world_size() > 1

class ConfidenceBasedDataSelector:
    """Select data point based on confidence"""
    def __init__(self, confidence_ratio=0.1, category_names=()):
        self.confidence_ratio = confidence_ratio
        self.categories = []
        self.scores = []
        self.category_names = category_names
        self.per_category_thresholds = None
        print('>>>>>>>>')
        print('confidence_ratio {}'.format(confidence_ratio))

    def extend(self, categories, scores):
        self.categories.extend(categories)
        self.scores.extend(scores)

    def calculate(self):
        per_category_scores = {c: [] for c in self.category_names}
        for c, s in zip(self.categories, self.scores):
            per_category_scores[c].append(s)

        per_category_thresholds = {}
        print(per_category_scores.keys())
        for c, s in per_category_scores.items():
            s.sort(reverse=True)    #降序
            print(c, len(s), int(self.confidence_ratio * len(s)))
            # confidence_ratio按照score降序取百分比
            per_category_thresholds[c] = s[int(self.confidence_ratio * len(s))] if len(s) else 1.

        print('----------------------------------------------------')
        print("confidence threshold for each category:")
        for c in self.category_names:
            print('\t', c, round(per_category_thresholds[c], 3))
        print('----------------------------------------------------')

        self.per_category_thresholds = per_category_thresholds

    def whether_select(self, categories, scores):
        assert self.per_category_thresholds is not None, "please call calculate before selection!"
        return [s > self.per_category_thresholds[c] for c, s in zip(categories, scores)]


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy that's robust to label noise"""
    def __init__(self, *args, offset=0.1, **kwargs):
        self.offset = offset
        super(RobustCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(torch.clamp(input + self.offset, max=1.), target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='sum') / input.shape[0]


class CategoryAdaptor:
    def __init__(self, class_names, log, args):
        self.class_names = class_names
        for k, v in args._get_kwargs():
            setattr(args, k.rstrip("_c"), v)
        self.args = args
        # print(self.args)
        self.logger = CompleteLogger(log)
        self.selector = ConfidenceBasedDataSelector(self.args.confidence_ratio, range(len(self.class_names) + 1))

        # create model
        print("=> using model '{}'".format(args.arch))
        backbone = utils.get_model(args.arch, pretrain=not args.scratch)
        pool_layer = nn.Identity() if args.no_pool else None
        num_classes = len(self.class_names) + 1
        self.model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                     pool_layer=pool_layer, finetune=not args.scratch).to(device)

    def load_checkpoint(self, name='latest'):
        if osp.exists(self.logger.get_checkpoint_path(name)):
            checkpoint = torch.load(self.logger.get_checkpoint_path(name), map_location='cpu')
            self.model.load_state_dict(checkpoint)
            return True
        else:
            return False

    def prepare_training_data(self, proposal_list: List[Proposal], labeled=True, distributed=False):
        if not labeled:
            # remove proposals with confidence score between (ignored_scores[0], ignored_scores[1])
            filtered_proposals_list = []
            assert len(self.args.ignored_scores) == 2 and self.args.ignored_scores[0] <= self.args.ignored_scores[1], \
                "Please provide a range for ignored_scores!"
            for proposals in proposal_list:
                keep_indices = ~((self.args.ignored_scores[0] < proposals.pred_scores)
                                 & (proposals.pred_scores < self.args.ignored_scores[1]))
                filtered_proposals_list.append(proposals[keep_indices])

            # calculate confidence threshold for each cateogry on the target domain
            for proposals in filtered_proposals_list:
                self.selector.extend(proposals.pred_classes.tolist(), proposals.pred_scores.tolist())
            self.selector.calculate()
        else:
            # remove proposals with ignored classes or ious between (ignored_ious[0], ignored_ious[1])
            filtered_proposals_list = []
            for proposals in proposal_list:
                keep_indices = (proposals.gt_classes != -1) & \
                               ~((self.args.ignored_ious[0] < proposals.gt_ious) &
                                 (proposals.gt_ious < self.args.ignored_ious[1]))
                filtered_proposals_list.append(proposals[keep_indices])

        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_train)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            ResizeImage(self.args.resize_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
            T.RandomGrayscale(),
            T.ToTensor(),
            normalize
        ])

        dataset = ProposalDataset(filtered_proposals_list, transform)
        if distributed:
            print('distribute dataset ......')
            self.train_sampler = DistributedSampler(dataset, drop_last=True)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                sampler=self.train_sampler, num_workers=self.args.workers, drop_last=True)
        else:
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=True, num_workers=self.args.workers, drop_last=True)
        # print('>>>>> one dataset, dataset_length:{}, batch_size:{}, dataloader_length:{}'.format(
        #     len(dataset), self.args.batch_size, len(dataloader)
        # ))
        return dataloader

    def prepare_validation_data(self, proposal_list: List[Proposal], distributed=False):
        """call this function if you have labeled data for validation"""
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            ResizeImage(self.args.resize_size),
            T.ToTensor(),
            normalize
        ])

        # remove proposals with ignored classes
        filtered_proposals_list = []
        for proposals in proposal_list:
            keep_indices = proposals.gt_classes != -1
            filtered_proposals_list.append(proposals[keep_indices])

        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_val)
        dataset = ProposalDataset(filtered_proposals_list, transform)
        # dataset = ProposalDatasetTest(filtered_proposals_list, transform)   # !!!!!
        if distributed:
            sampler = DistributedSampler(dataset, drop_last=False)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                sampler=sampler, num_workers=self.args.workers, drop_last=False)
        else:
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        # print('>>>>> one dataset, dataset_length:{}, batch_size:{}, dataloader_length:{}'.format(
        #     len(dataset), self.args.batch_size, len(dataloader)))
        return dataloader

    def prepare_test_data(self, proposal_list: List[Proposal], distributed=False):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            ResizeImage(self.args.resize_size),
            T.ToTensor(),
            normalize
        ])

        dataset = ProposalDataset(proposal_list, transform)
        if distributed:
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                sampler=sampler, num_workers=self.args.workers)
        else:
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        return dataloader

    def fit(self, data_loader_source, data_loader_target, data_loader_validation=None, data_loader_test=None, distributed=False):
        """When no labels exists on target domain, please set data_loader_validation=None"""
        args = self.args
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        cudnn.benchmark = True

        iter_source = ForeverDataIterator(data_loader_source)
        iter_target = ForeverDataIterator(data_loader_target)

        model = self.model
        feature_dim = model.features_dim
        num_classes = len(self.class_names) + 1

        if args.randomized:
            domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
        else:
            domain_discri = DomainDiscriminator(feature_dim * num_classes, hidden_size=1024).to(device)

        all_parameters = model.get_parameters() + domain_discri.get_parameters()
        # define optimizer and lr scheduler
        optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

        # define loss function
        domain_adv = ConditionalDomainAdversarialLoss(
            domain_discri, entropy_conditioning=args.entropy,
            num_classes=num_classes, features_dim=feature_dim, randomized=args.randomized,
            randomized_dim=args.randomized_dim
        ).to(device)

        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )
            domain_adv = DistributedDataParallel(
                domain_adv, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )

        # start training
        best_acc1_target = 0.
        best_acc1_test = 0.
        best_epoch = 0
        num_iters_source = len(iter_source)
        num_iters_target = len(iter_target)
        print('category fitting ......')
        print('num_iters_source: {}'.format(num_iters_source))
        print('num_iters_target: {}'.format(num_iters_target))
        if data_loader_validation is not None:
            print('num_iters_validation: {}'.format(len(data_loader_validation)))
        if data_loader_test is not None:
            print('num_iters_test: {}'.format(len(data_loader_test)))

        num_gpus = comm.get_world_size()
        print('num_gpus: {}'.format(num_gpus))
        if args.debug:
            args.iters_per_epoch = 21
        elif args.iters_perepoch_mode == 'compute_from_epoch':
            args.iters_per_epoch = int(max(num_iters_source, num_iters_target) * args.coefficient)
        standard_iters_per_epoch_one_gpu = 1000 * 64 // data_loader_source.batch_size
        if args.iters_per_epoch > standard_iters_per_epoch_one_gpu // num_gpus:
            args.iters_per_epoch = standard_iters_per_epoch_one_gpu  // num_gpus
        print('num iters per epoch:', args.iters_per_epoch)

        for epoch in range(args.epochs):
            print("lr:", lr_scheduler.get_last_lr()[0])
            # train for one epoch
            batch_time = AverageMeter('Time', ':3.1f')
            data_time = AverageMeter('Data', ':3.1f')
            losses = AverageMeter('Loss', ':3.2f')
            losses_t = AverageMeter('Loss(t)', ':3.2f')
            trans_losses = AverageMeter('Trans Loss', ':3.2f')
            cls_accs = AverageMeter('Cls Acc', ':3.1f')
            domain_accs = AverageMeter('Domain Acc', ':3.1f')
            progress = ProgressMeter(
                args.iters_per_epoch,
                [batch_time, data_time, losses, losses_t, trans_losses, cls_accs, domain_accs],
                prefix="Epoch: [{}]".format(epoch))

            # switch to train mode
            model.train()
            domain_adv.train()

            end = time.time()
            print_end = time.time()
            
            # print('data_loader_target: {}'.format(len(data_loader_target)))
            # for i, data in enumerate(data_loader_target):
            #     x_t, label = data
            #     x_t = x_t.to(device)
            #     print('iter: {}. {} {}'.format(i, x_t.shape, x_t.device))

            self.train_sampler.set_epoch(epoch)

            for i in range(args.iters_per_epoch):
                x_s, labels_s = next(iter_source)
                x_t, labels_t = next(iter_target)

                # assign pseudo labels for target-domain proposals with extremely high confidence
                # 根据提前计算的confidence-ratio计算出的可选择置信度阈值，来选取当前target batch中的可用样本
                selected = torch.tensor(
                    self.selector.whether_select(
                        labels_t['pred_classes'].numpy().tolist(),
                        labels_t['pred_scores'].numpy().tolist()
                    )
                )
                # 将小置信度的样本设置为-1类背景
                pseudo_classes_t = selected * labels_t['pred_classes'] + (~selected) * -1
                pseudo_classes_t = pseudo_classes_t.to(device)

                x_s = x_s.to(device)
                x_t = x_t.to(device)
                gt_classes_s = labels_s['gt_classes'].to(device)
                # print('iter: {}. {} {}'.format(i, x_t.shape, x_t.device))
                # print('iter: {}'.format(i))
                # print(x_s.shape, x_t.shape)
                # print(x_s.shape ==  x_t.shape)
                # continue

                # measure data loading time
                one_data_time = time.time()
                data_time.update(one_data_time - end)
                # print('device_time: {:.3f}'.format(one_data_time - end))

                # compute output
                x = torch.cat((x_s, x_t), dim=0)    #此处相当于batchsize变大两倍，可以不cat，使bs与设置保持一直
                # pre_time = time.time()
                # print('pre_time: {:.3f}'.format(pre_time - one_data_time ))
                y, f = model(x)
                # model_time = time.time()
                # print('model_time: {:.3f}'.format(model_time - pre_time))
                y_s, y_t = y.chunk(2, dim=0)
                f_s, f_t = f.chunk(2, dim=0)
                # post_time = time.time()
                # print('post_time: {:.3f}'.format(post_time - model_time))

                cls_loss = F.cross_entropy(y_s, gt_classes_s, ignore_index=-1)
                cls_loss_t = RobustCrossEntropyLoss(ignore_index=-1, offset=args.epsilon)(y_t, pseudo_classes_t)
                transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
                # domain_acc = domain_adv.domain_discriminator_accuracy
                loss = cls_loss + transfer_loss * args.trade_off + cls_loss_t

                cls_acc = accuracy(y_s, gt_classes_s)[0]

                losses.update(loss.item(), x_s.size(0))
                cls_accs.update(cls_acc, x_s.size(0))
                # domain_accs.update(domain_acc, x_s.size(0))
                trans_losses.update(transfer_loss.item(), x_s.size(0))
                losses_t.update(cls_loss_t.item(), x_s.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and i != 0 and comm.is_main_process():
                    progress.display(i)
                    print_time = time.time() - print_end
                    current_time = time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))
                    print('{} Print time per {} iter: {:.3f} Seconds  |  Estimated complete time: {:.1f} Mins\n'.format(
                        current_time, args.print_freq, print_time, print_time / args.print_freq * args.iters_per_epoch / 60))
                    print_end = time.time()

            # evaluate on validation set
            if data_loader_validation is not None:
                acc1 = self.validate(data_loader_validation, model, self.class_names, args)
                if acc1 > best_acc1_target:
                    if comm.is_main_process():
                        torch.save(model.state_dict(), self.logger.get_checkpoint_path('best_target'))
                        print('best acc at epoch {} update to {}'.format(epoch, acc1))
                    best_epoch = epoch
                    best_acc1_target = acc1
            if data_loader_test is not None:
                acc1 = self.validate(data_loader_test, model, self.class_names, args)
                if acc1 > best_acc1_test:
                    if comm.is_main_process():
                        torch.save(model.state_dict(), self.logger.get_checkpoint_path('best_test'))
                        print('best acc at epoch {} update to {}'.format(epoch, acc1))
                    best_epoch = epoch
                    best_acc1_test = acc1
                

            # save checkpoint
            if comm.is_main_process():
                torch.save(model.state_dict(), self.logger.get_checkpoint_path('latest'))

        if comm.is_main_process():
            print("best_acc1 = {:3.1f} ad epoch {}".format(best_acc1_target, best_epoch))
            print("best_acc1_test = {:3.1f} ad epoch {}".format(best_acc1_test, best_epoch))
        domain_adv.to(torch.device("cpu"))
        self.logger.logger.flush()

    def predict(self, data_loader):
        if distributed:
            self.model = DistributedDataParallel(
                self.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )
        # switch to evaluate mode
        self.model.eval()
        predictions = deque()
        # print(self.model.device)

        with torch.no_grad():
            end_time = time.time()
            for images, _ in tqdm.tqdm(data_loader):
                data_time = time.time()
                # print('data time: {:.3f}'.format(data_time- end_time))
                images = images.to(device)
                device_time = time.time()
                # print('device_time: {:.3f}'.format(device_time- data_time))
                # compute output
                output = self.model(images)
                model_time = time.time()
                # print('model_time: {:.3f}'.format(model_time - device_time))
                prediction = output.argmax(-1).cpu().numpy().tolist()
                for p in prediction:
                    predictions.append(p)
                post_time = time.time()
                # print('post_time: {:.3f}'.format(post_time - model_time))
                end_time = time.time()
        return predictions

    @staticmethod
    def validate(val_loader, model, class_names, args) -> float:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()
        confmat = ConfusionMatrix(len(class_names)+1)
        pred_class_dict = {}
        gt_dict = {}
        pred_score_dict = {}
        for class_name in class_names + ['bg']:
            pred_score_dict[class_name] = []
            pred_class_dict[class_name] = []
            gt_dict[class_name] = []

        pred_class_ls = []
        gt_ls = []
        pred_score_ls = []

        with torch.no_grad():
            end = time.time()
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                gt_classes = labels['gt_classes'].to(device)

                # compute output
                output = model(images)
                loss = F.cross_entropy(output, gt_classes)

                # measure accuracy and record loss
                acc1, = accuracy(output, gt_classes, topk=(1,))
                confmat.update(gt_classes, output.argmax(1))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))

                gt_ls.extend(gt_classes.to(torch.device("cpu")).numpy())
                pred_class_ls.extend(output.cpu().argmax(1).numpy())
                pred_score_ls.extend(output.cpu().max(1)[0].numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and comm.is_main_process():
                    progress.display(i)
            if comm.is_main_process():
                print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
                print(confmat.format(class_names+["bg"]))

            score_thresholds=[0, 0.6, 0.9, 0.95, 0.97, 0.98, 0.99]
            confusion_res = compute_confusionmatrix(gt_ls, pred_class_ls, pred_score_ls, \
                class_names+['bg'], score_thresholds=score_thresholds)
            table = prettytable.PrettyTable()
            table.add_column('class', class_names + ['bg'])
            for score_threshold in score_thresholds:
                one_confusion_res = confusion_res[str(score_threshold)]
                acc = one_confusion_res['acc']
                recall = one_confusion_res['recall']
                # print("PR result at score_threshold {}".format(score_threshold))
                # table = prettytable.PrettyTable(["class", "acc", "recall"])
                write_column_name = 's_{} P/R'.format(score_threshold)
                write_column = []
                for i, per_class, per_acc, per_recall in zip(range(len(class_names+['bg'])), class_names+['bg'], (acc * 100).tolist(), (recall * 100).tolist()):
                    write_column += ['{:.1f}  {:.1f}'.format(per_acc, per_recall)]
                table.add_column(write_column_name, write_column)
            if comm.is_main_process():
                print(table.get_string())

        return top1.avg

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        # dataset parameters
        parser.add_argument('--resize-size-c', type=int, default=112,
                            help='the image size after resizing')
        parser.add_argument('--ignored-scores-c', type=float, nargs='+', default=[0.05, 0.3])
        parser.add_argument('--max-train-c', type=int, default=10)
        parser.add_argument('--max-val-c', type=int, default=2)
        parser.add_argument('--ignored-ious-c', type=float, nargs='+', default=(0.4, 0.5),
                            help='the iou threshold for ignored boxes')
        # model parameters
        parser.add_argument('--arch-c', metavar='ARCH', default='resnet101',
                            choices=utils.get_model_names(),
                            help='backbone architecture: ' +
                                 ' | '.join(utils.get_model_names()) +
                                 ' (default: resnet101)')
        # parser.add_argument('--arch-c', metavar='ARCH', default='resnet50',
        #                     choices=utils.get_model_names(),
        #                     help='backbone architecture: ' +
        #                          ' | '.join(utils.get_model_names()) +
        #                          ' (default: resnet101)')
        parser.add_argument('--debug', action='store_true',
                            help='debug.')

        parser.add_argument('--bottleneck-dim-c', default=1024, type=int,
                            help='Dimension of bottleneck')
        parser.add_argument('--no-pool-c', action='store_true',
                            help='no pool layer after the feature extractor.')
        parser.add_argument('--scratch-c', action='store_true', help='whether train from scratch.')
        parser.add_argument('--randomized-c', action='store_true',
                            help='using randomized multi-linear-map (default: False)')
        parser.add_argument('--randomized-dim-c', default=1024, type=int,
                            help='randomized dimension when using randomized multi-linear-map (default: 1024)')
        parser.add_argument('--entropy-c', default=False, action='store_true', help='use entropy conditioning')
        parser.add_argument('--trade-off-c', default=1., type=float,
                            help='the trade-off hyper-parameter for transfer loss')
        parser.add_argument('--confidence-ratio-c', default=0.0, type=float)
        parser.add_argument('--epsilon-c', default=0.01, type=float,
                            help='epsilon hyper-parameter in Robust Cross Entropy')
        # training parameters
        parser.add_argument('--batch-size-c', default=64, type=int,
                            metavar='N',
                            help='mini-batch size (default: 64)')   #96
                            # 64，实际预测时，会把source和target cat到一起，batch_size会变成2倍
        parser.add_argument('--learning-rate-c', default=0.01, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-gamma-c', default=0.001, type=float, help='parameter for lr scheduler')
        parser.add_argument('--lr-decay-c', default=0.75, type=float, help='parameter for lr scheduler')
        parser.add_argument('--momentum-c', default=0.9, type=float, metavar='M', help='momentum')
        parser.add_argument('--weight-decay-c', default=1e-3, type=float,
                            metavar='W', help='weight decay (default: 1e-3)',
                            dest='weight_decay')
        parser.add_argument('--workers-c', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 2)')
        parser.add_argument('--epochs-c', default=6, type=int, metavar='N',
                            help='number of total epochs to run')   # 10
        parser.add_argument('--iters-per-epoch-c', default=1000, type=int,
                            help='Number of iterations per epoch')
        parser.add_argument('--iters-perepoch-mode', default='compute_from_epoch', type=str,
                            help='iters-perepoch-mode')    
        parser.add_argument('--coefficient', default=1, type=int,
                            help='iters-perepoch-mode coefficient')                   
        parser.add_argument('--print-freq-c', default=10, type=int,
                            metavar='N', help='print frequency (default: 100)')
        parser.add_argument('--seed-c', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument("--log-c", type=str, default='cdan',
                            help="Where to save logs, checkpoints and debugging images.")
        return parser
