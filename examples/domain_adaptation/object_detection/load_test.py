import json
# path = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_adapt/logs/faster_rcnn_R_101_C4/cityscapes2foggy_4/phase1/cache/proposal/.._datasets_cityscapes_in_voc_trainval_bg.json'
path = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/cascade_adapt/logs/faster_rcnn_R_101_C4/cityscapes2foggy_4/phase1/cache/proposal/.._datasets_cityscapes_in_voc_trainval_fg.json'

with open(path, 'r') as f:
    a = json.load(f)
