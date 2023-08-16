#!/bin/bash

python -u weaver/train.py \
 --gpu "0,1" \
 --train-mode regression \
 --batch-size 768 --start-lr 6.75e-4 --num-epochs 3000 --optimizer ranger \
 --data-train '/fmhwwvol/ntuples/*/train/*.root' \
 --data-test '/fmhwwvol/ntuples/*/test/*.root' \
 --data-config weaver/data_new/finetune/FM_ak8_mass_regression_v10.yaml \
 --network-config weaver/networks/disco_test/mlp_2p_gated_regression_disco.py \
 --model-prefix /fmhwwvol/mq_experiments/mass_regression_disco_v1/model \
 --log-file /fmhwwvol/mq_experiments/mass_regression_disco_v1/logs/train.log \
 --predict-output /fmhwwvol/mq_experiments/mass_regression_disco_v1/predict/pred.root \
 --num-workers 0 \
 --in-memory \
 --steps-per-epoch=1 \
 --load-model-weights finetune_gghww_custom \
 --discolambda 100.0 \
 --discovar fj_ParT_inclusive_score \
 
mv runs/* /fmhwwvol/runs_melissa/
