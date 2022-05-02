# Faster RCNN: VOC->Clipart
CUDA_VISIBLE_DEVICES=2,3 python source_only.py --num-gpus 2 \
  --config-file config/faster_rcnn_R_101_C4_voc_hold_gpu.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/source_only_hold_gpu/faster_rcnn_R_101_C4/voc2clipart \
  TEST.EVAL_PERIOD 1000000000000
