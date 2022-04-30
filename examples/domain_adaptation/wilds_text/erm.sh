# civilcomments
CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds -d "civilcomments" --lr 1e-05 --deterministic \
    --log logs/erm/civilcomments --unlabeled-list "extra_unlabeled" --metric "acc_wg" --seed 0 \
    --max-token-length 300 --wd 0.01 --uniform-over-groups --groupby-fields y black

# amazon
CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds -d "amazon" --log logs/erm/amazon -b 24 24 \
    --metric "10th_percentile_acc" --lr 1e-5 --max-token-length 512 --wd 0.01 --seed 1 --epochs 3 --deterministic