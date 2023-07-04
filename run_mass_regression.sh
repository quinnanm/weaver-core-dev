python weaver/train.py \
 --gpus '' \
 --train-mode regression \
 --batch-size 768 --start-lr 6.75e-3 --num-epochs 1 --optimizer ranger \
 --data-train 'datafiles/ntuples/*/out.root' \
 --data-test 'datafiles/ntuples/*/out.root' \
 --data-config weaver/data_new/finetune/FM_ak8_mass_regression.yaml \
 --network-config weaver/networks/fintune_test/mlp_2p_gated_regression.py \
 --model-prefix experiments/mass_regression/ \
 --log-file experiments/mass_regression/logs/train.log \
 --predict-output experiments/mass_regression/predict/pred.root \
 --num-workers=0
#  --tensorboard mass_regression