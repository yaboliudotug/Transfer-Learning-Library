"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import argparse
import os
import shutil
import time
import pprint
import math
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
import wilds

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier as Classifier
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter
from tllib.utils.metric import accuracy


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    writer = SummaryWriter(args.log)
    pprint.pprint(args)

    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    # Data loading code
    train_transform = utils.get_train_transform(
        img_size=args.img_size,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.interpolation,
    )
    val_transform = utils.get_val_transform(
        img_size=args.img_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
    )
    if args.local_rank == 0:
        print("train_transform: ", train_transform)
        print("val_transform: ", val_transform)

    train_labeled_dataset, train_unlabeled_dataset, test_datasets, args.num_classes, args.class_names = \
        utils.get_dataset(args.data, args.data_dir, args.unlabeled_list, args.test_list,
                          train_transform, val_transform, verbose=args.local_rank == 0)

    # create model
    if args.local_rank == 0:
        if not args.scratch:
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    model = Classifier(backbone, args.num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch)
    features_dim = model.features_dim
    domain_discri = DomainDiscriminator(features_dim, hidden_size=1024, sigmoid=False)

    if args.sync_bn:
        import apex
        if args.local_rank == 0:
            print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)
    domain_discri = domain_discri.cuda().to(memory_format=memory_format)

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size[0] * args.world_size) / 256.
    optimizer = torch.optim.SGD(
        model.get_parameters() + domain_discri.get_parameters(), args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    (model, domain_discri), optimizer = amp.initialize([model, domain_discri], optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    # Use cosine annealing learning rate strategy
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: max((math.cos(float(x) / args.epochs * math.pi) * 0.5 + 0.5) * args.lr, args.min_lr)
    )

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri, sigmoid=False)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)
        domain_adv = DDP(domain_adv, delay_allreduce=True)

    # define loss function (criterion)
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(args.smoothing).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    # Data loading code
    train_labeled_sampler = None
    train_unlabeled_sampler = None
    if args.distributed:
        train_labeled_sampler = DistributedSampler(train_labeled_dataset)
        train_unlabeled_sampler = DistributedSampler(train_unlabeled_dataset)

    train_labeled_loader = DataLoader(
        train_labeled_dataset, batch_size=args.batch_size[0], shuffle=(train_labeled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_labeled_sampler, drop_last=True)
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset, batch_size=args.batch_size[1], shuffle=(train_unlabeled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_unlabeled_sampler, drop_last=True)

    if args.phase == 'test':
        # resume from the latest checkpoint
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)
        for n, d in zip(args.test_list, test_datasets):
            if args.local_rank == 0:
                print(n)
            utils.validate(d, model, -1, writer, args)
        return

    for epoch in range(args.epochs):
        if args.distributed:
            train_labeled_sampler.set_epoch(epoch)
            train_unlabeled_sampler.set_epoch(epoch)

        lr_scheduler.step(epoch)
        if args.local_rank == 0:
            print(lr_scheduler.get_last_lr())
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[-1], epoch)
        # train for one epoch
        train(train_labeled_loader, train_unlabeled_loader, model, criterion, domain_adv, optimizer, epoch, writer, args)

        # evaluate on validation set
        for n, d in zip(args.test_list, test_datasets):
            if args.local_rank == 0:
                print(n)
            prec1 = utils.validate(d, model, epoch, writer, args)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
            if is_best:
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

    logger.close()
    writer.close()


def train(train_labeled_loader, train_unlabeled_loader, model, criterion, domain_adv,
          optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':3.1f')
    losses_s = AverageMeter('Loss (s)', ':3.2f')
    losses_trans = AverageMeter('Loss (transfer)', ':3.2f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    top1 = AverageMeter('Top 1', ':3.1f')

    # switch to train mode
    model.train()
    end = time.time()

    num_iterations = min(len(train_labeled_loader), len(train_unlabeled_loader))

    for i, (input_s, target_s, metadata_s), (input_t, metadata_t) in \
            zip(range(num_iterations), train_labeled_loader, cycle(train_unlabeled_loader)):

        # compute output
        n_s, n_t = len(input_s), len(input_t)
        input = torch.cat([input_s.cuda(), input_t.cuda()], dim=0)
        output, feature = model(input)
        output_s, output_t = output.split([n_s, n_t], dim=0)
        feature_s, feature_t = feature.split([n_s, n_t], dim=0)
        loss_s = criterion(output_s, target_s.cuda())
        loss_trans = domain_adv(feature_s, feature_t)
        loss = loss_s + loss_trans * args.trade_off

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, = accuracy(output_s.data, target_s.cuda(), topk=(1,))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss_s = utils.reduce_tensor(loss_s.data, args.world_size)
                reduced_loss_trans = utils.reduce_tensor(loss_trans.data, args.world_size)
                prec1 = utils.reduce_tensor(prec1, args.world_size)
                domain_acc = domain_adv.module.domain_discriminator_accuracy
            else:
                reduced_loss_s = loss_s.data
                reduced_loss_trans = loss_trans.data
                domain_acc = domain_adv.domain_discriminator_accuracy

            # to_python_float incurs a host<->device sync
            losses_s.update(to_python_float(reduced_loss_s), input_s.size(0))
            losses_trans.update(to_python_float(reduced_loss_trans), input_s.size(0))
            domain_accs.update(to_python_float(domain_acc), input_s.size(0))
            top1.update(to_python_float(prec1), input_s.size(0))
            global_step = epoch * num_iterations + i

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                writer.add_scalar('train/top1', to_python_float(prec1), global_step)
                writer.add_scalar("train/loss (s)", to_python_float(reduced_loss_s), global_step)
                writer.add_scalar("train/loss (trans)", to_python_float(reduced_loss_trans), global_step)
                writer.add_figure('train/predictions vs. actuals',
                                  utils.plot_classes_preds(input_s.cpu(), target_s, output_s.cpu(), args.class_names,
                                                           metadata_s, train_labeled_loader.dataset.metadata_map),
                                  global_step=global_step)

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss (s) {loss_s.val:.10f} ({loss_s.avg:.4f})\t'
                      'Loss (trans) {loss_trans.val:.10f} ({loss_trans.avg:.4f})\t'
                      'Domain Acc {domain_acc.val:.10f} ({domain_acc.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_labeled_loader),
                    args.world_size * args.batch_size[0] / batch_time.val,
                    args.world_size * args.batch_size[0] / batch_time.avg,
                    batch_time=batch_time, loss_s=losses_s, loss_trans=losses_trans,
                    domain_acc=domain_accs, top1=top1))


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='fmow', choices=wilds.supported_datasets,
                        help='dataset: ' + ' | '.join(wilds.supported_datasets) +
                             ' (default: fmow)')
    parser.add_argument('--unlabeled-list', nargs='+', default=["test_unlabeled", ])
    parser.add_argument('--test-list', nargs='+', default=["val", "test"])
    parser.add_argument('--metric', default="acc_worst_region")
    parser.add_argument('--img-size', type=int, default=(224, 224), metavar='N', nargs='+',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--crop-pct', default=utils.DEFAULT_CROP_PCT, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default='bicubic', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.5, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)')
    # model parameters
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--bottleneck-dim', default=512, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # Learning rate schedule parameters
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR',
                        help='Initial learning rate.  Will be scaled by <global batch size>/256: '
                             'args.lr = args.lr*float(args.batch_size*args.world_size)/256.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # training parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=(64, 64), type=int, nargs='+',
                        metavar='N', help='mini-batch size per process for source'
                                          ' and target domain (default: (64, 64))')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    args = parser.parse_args()
    main(args)

