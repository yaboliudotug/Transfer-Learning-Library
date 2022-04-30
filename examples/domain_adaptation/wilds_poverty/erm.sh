# official split scheme
CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds --arch 'resnet18_ms' \
    --opt-level O1 --deterministic --log logs/erm/poverty --lr 1e-3 --wd 0.0  \
    --epochs 200 --metric r_wg --split_scheme official -b 64 64 --fold A

CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds --arch 'resnet18_ms' \
    --opt-level O1 --deterministic --log logs/erm/poverty --lr 1e-3 --wd 0.0  \
    --epochs 200 --metric r_wg --split_scheme official -b 64 64 --fold B

CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds --arch 'resnet18_ms' \
    --opt-level O1 --deterministic --log logs/erm/poverty --lr 1e-3 --wd 0.0  \
    --epochs 200 --metric r_wg --split_scheme official -b 64 64 --fold C

CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds --arch 'resnet18_ms' \
    --opt-level O1 --deterministic --log logs/erm/poverty --lr 1e-3 --wd 0.0  \
    --epochs 200 --metric r_wg --split_scheme official -b 64 64 --fold D

CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds --arch 'resnet18_ms' \
    --opt-level O1 --deterministic --log logs/erm/poverty --lr 1e-3 --wd 0.0  \
    --epochs 200 --metric r_wg --split_scheme official -b 64 64 --fold E