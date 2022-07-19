
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_oriinference2/model_0027999.pth
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_oriinference3/model_0041999.pth
# CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  \
#   --config-file config/faster_rcnn_R_101_C4_cityscapes_iou.yaml --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_test_high1/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
  --use-pre-cache 1 --use-pre-crop 0 --update-score 0 --update-proposal 0 \
  --num-cascade 1 --cascade-flag-category 1 --cascade-flag-bbox 0 \
  --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_debug_fgness_updatecls/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=1 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config/faster_rcnn_R_101_C4_cityscapes_iou.yaml --use-pre-cache 1 --use-pre-crop 0 --update-score 0 --update-proposal 0 \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_debug_fgness_use-prediou/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0
