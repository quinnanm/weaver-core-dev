#!/bin/bash

python -u weaver/train.py \
 --gpu "0,1" \
 --train-mode regression \
 --batch-size 768 --start-lr 6.75e-4 --num-epochs 3000 --optimizer ranger \
 --data-train '/fmhwwvol/ntuples/TTToSemiLeptonic/train/*.root' \
 --data-test '/fmhwwvol/ntuples/*/test/*.root' \
 --data-config weaver/data_new/finetune/FM_ak8_mass_regression_v1.yaml \
 --network-config weaver/networks/fintune_test/mlp_2p_gated_regression.py \
 --model-prefix /fmhwwvol/experiments_sara/mass_regression_v1/model \
 --log-file /fmhwwvol/experiments_sara/mass_regression_v1/logs/train.log \
 --predict-output /fmhwwvol/experiments_sara/mass_regression_v1/predict/pred.root \
 --num-workers 0 \
 --in-memory \
 --steps-per-epoch=1 \
 --tensorboard _v1 \
 --load-model-weights finetune_gghww_custom \

mkdir -p /fmhwwvol/runs_sara/
mv runs/* /fmhwwvol/runs_sara/
