import time
from torchvision.datasets.folder import default_loader

big_img_path = '/disk/liuyabo/data/cityscapes/leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png'
small_img_path = '/disk/liuyabo/research/Transfer-Learning-Library/examples/domain_adaptation/object_detection/crop_img.jpg'

for i in range(100):
    time1 = time.time()
    img = default_loader(big_img_path)
    time2 = time.time()
    img = default_loader(small_img_path)
    time3 = time.time()
    print('read time: {:.3f},  {:.3f}'.format(time2 - time1, time3 - time2))
