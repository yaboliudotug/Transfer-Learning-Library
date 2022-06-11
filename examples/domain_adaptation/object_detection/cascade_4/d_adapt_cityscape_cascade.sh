
# # ResNet101 Based Faster RCNN: Cityscapes -> Foggy Cityscapes
# # 40.1
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/cityscapes2foggy_cache/model_0029999.pth
CUDA_VISIBLE_DEVICES=2 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  --use-pre-cache 1 --num-cascade 3 --cascade-flag-category 111 --cascade-flag-bbox 000 \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_cascade_test_res50/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0


# pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/cityscapes2foggy_cache/model_0029999.pth
# CUDA_VISIBLE_DEVICES=3 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --use-pre-cache 1 --update-score 1\
#   --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml --num-cascade 3 --cascade-flag-category 111 --cascade-flag-bbox 001 \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_cascade_test2_updatescore/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

#  --confidence-ratio-c 0.05
# # 42.4
# pretrained_models=logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase1/model_final.pth
# CUDA_VISIBLE_DEVICES=2 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.1 \
#   --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0
