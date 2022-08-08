# #train_fgness_ori
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 0 --update-score 0 --update-proposal 0 \
#   --num-cascade 0 --cascade-flag-category 1 --cascade-flag-bbox 0 \
#   --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_ori/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# #train_fgness_updatecls_num1
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=1 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   --num-cascade 1 --cascade-flag-category 1 --cascade-flag-bbox 0 \
#   --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num1/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# #train_fgness_updatecls_num2
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   --num-cascade 2 --cascade-flag-category 11 --cascade-flag-bbox 00 \
#   --ignored-scores-ls [[0.05,0.5],[0.05,0.5]] --ignored-ious-ls [[0.4,0.5],[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num2/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# train_fgness_updatecls_num3
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=1 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   --num-cascade 3 --cascade-flag-category 111 --cascade-flag-bbox 000 \
#   --ignored-scores-ls [[0.05,0.5],[0.05,0.5],[0.05,0.5]] --ignored-ious-ls [[0.4,0.5],[0.4,0.5],[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num3/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# #train_fgness_updatecls_num1_filter0208
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=1 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   --num-cascade 1 --cascade-flag-category 1 --cascade-flag-bbox 0 \
#   --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num1_filter0208/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# #train_fgness_updatecls_num1_clsbox
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   --num-cascade 1 --cascade-flag-category 1 --cascade-flag-bbox 1 \
#   --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num1_clsbox/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0


# #train_fgness_updatecls_num1_filter0508
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=1 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   --num-cascade 1 --cascade-flag-category 1 --cascade-flag-bbox 0 \
#   --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num1_filter0508/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# #train_fgness_updatecls_num2_clsbox_updatesb
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 1 --update-proposal 1 \
#   --num-cascade 2 --cascade-flag-category 11 --cascade-flag-bbox 11 \
#   --ignored-scores-ls [[0.05,0.5],[0.05,0.5]] --ignored-ious-ls [[0.4,0.5],[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num2_clsbox_updatesb/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# #train_fgness_updatecls_num1_v2
# pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
# CUDA_VISIBLE_DEVICES=1 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
#   --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
#   --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
#   --num-cascade 1 --cascade-flag-category 1 --cascade-flag-bbox 0 \
#   --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
#   -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
#   --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#   OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num1_v2/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# #train_fgness_updatecls_num1_iou
pretrained_models=../logs/source_only_iou/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_fg_split/model_0013999.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5  --confidence-ratio-c 0.1 \
  --config-file config_iou/faster_rcnn_R_101_C4_cityscapes_iou.yaml \
  --use-pre-cache 1 --use-pre-crop 1 --update-score 0 --update-proposal 0 \
  --num-cascade 1 --cascade-flag-category 1 --cascade-flag-bbox 0 \
  --ignored-scores-ls [[0.05,0.5]] --ignored-ious-ls [[0.4,0.5]] \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_fgness_updatecls_num1_iou/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0


