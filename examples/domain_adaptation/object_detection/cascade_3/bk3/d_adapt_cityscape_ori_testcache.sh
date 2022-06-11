# # ResNet101 Based Faster RCNN: Cityscapes -> Foggy Cityscapes
# # 40.1
# pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/cityscapes2foggy/model_final.pth
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/cityscapes2foggy_cache/model_0029999.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --num-gpus 1 --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml --use-pre-cache 1 \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy_ori_testcache/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

 #--confidence-ratio-c 0.05 \
