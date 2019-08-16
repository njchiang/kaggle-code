#!/bin/bash

conda activate pytorch
# cd ~/share/kaggle-code/OCT

python main.py \
    --data_dir=/opt/data/workingdir/nrakocz/DATA/kermany2018/OCT2017/OCT2017 \
    --gpu=0 \
    --save_dir=/opt/data/workingdir/jnchiang/share/oct/resnet50_for_doheny_visit-20190819 \
    --deploy  \
    --val \
    --restore \
    --checkpoint=/opt/data/workingdir/jnchiang/share/oct/weights/session-best.pt \
    --batch_size=8
    # --debug
