#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python erm.py data/food101 -d Food101 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/erm/food101_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/food101 -d Food101 --num-samples-per-class 10 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/erm/food101_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/food101 -d Food101 --oracle --finetune --lr 0.01 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/food101_oracle

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/cifar10_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/cifar10_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/cifar10_oracle

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/erm/cifar100_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 10 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/erm/cifar100_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --oracle --finetune --lr 0.01 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/cifar100_oracle

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm/cub200_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm/cub200_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 --oracle --finetune \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/cub200_oracle

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/aircraft_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/aircraft_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/aircraft_oracle

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python erm.py data/cars -d StanfordCars --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/car_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cars -d StanfordCars --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/car_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/cars -d StanfordCars --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/car_oracle

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python erm.py data/sun397 -d SUN397 --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/erm/sun_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/sun397 -d SUN397 --num-samples-per-class 10 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/erm/sun_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/sun397 -d SUN397 --oracle --finetune --lr 0.001 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/sun_oracle

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python erm.py data/dtd -d DTD --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/dtd_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/dtd -d DTD --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/dtd_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/dtd -d DTD --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/dtd_oracle

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python erm.py data/pets -d OxfordIIITPets --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/erm/pets_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/pets -d OxfordIIITPets --num-samples-per-class 10 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/erm/pets_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/pets -d OxfordIIITPets --oracle --finetune --lr 0.001 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/pets_oracle

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python erm.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/flowers_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/flowers -d OxfordFlowers102 --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/erm/flowers_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/flowers -d OxfordFlowers102 --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/flowers_oracle

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python erm.py data/caltech101 -d Caltech101 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm/caltech_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/caltech101 -d Caltech101 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm/caltech_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python erm.py data/caltech101 -d Caltech101 --oracle --finetune \
  -a resnet50 --epochs 80 --seed 0 --log logs/erm/caltech_oracle

# ImageNet Unsupervised Pretrain (MoCov2, ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python erm.py data/food101 -d Food101 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/food101_4_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/food101 -d Food101 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/food101_10_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/food101 -d Food101 --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/food101_oracle --lr 0.01 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cifar10_4_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cifar10_10_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cifar10_oracle --lr 0.001 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cifar100_4_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cifar100_10_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cifar100_oracle --lr 0.001 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cub200_4_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cub200_10_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cub200_oracle --lr 0.01 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/aircraft_4_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/aircraft_10_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/aircraft_oracle --lr 0.01 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python erm.py data/cars -d StanfordCars --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/car_4_labels_per_class --lr 0.03 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cars -d StanfordCars --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/car_10_labels_per_class --lr 0.03 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/cars -d StanfordCars --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/car_oracle --lr 0.03 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python erm.py data/sun397 -d SUN397 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/sun_4_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/sun397 -d SUN397 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/sun_10_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/sun397 -d SUN397 --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/sun_oracle --lr 0.001 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python erm.py data/dtd -d DTD --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/dtd_4_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/dtd -d DTD --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/dtd_10_labels_per_class --lr 0.001 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/dtd -d DTD --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/dtd_oracle --lr 0.001 -lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python erm.py data/pets -d OxfordIIITPets --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/pets_4_labels_per_class --lr 0.003 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/pets -d OxfordIIITPets --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/pets_10_labels_per_class --lr 0.003 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/pets -d OxfordIIITPets --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/pets_oracle --lr 0.003 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python erm.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/flowers_4_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/flowers -d OxfordFlowers102 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/flowers_10_labels_per_class --lr 0.01 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/flowers -d OxfordFlowers102 --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/flowers_oracle --lr 0.01 --lr-scheduler cos --epochs 40 -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python erm.py data/caltech101 -d Caltech101 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/caltech_4_labels_per_class --lr 0.003 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/caltech101 -d Caltech101 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/caltech_10_labels_per_class --lr 0.003 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python erm.py data/caltech101 -d Caltech101 --oracle --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/caltech_oracle --lr 0.003 --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
