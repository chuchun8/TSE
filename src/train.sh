# bash ./train.sh > train.log

####################################################################################################################
config=../config/config-bertweet.txt
train_data=../data/train.csv
dev_data=../data/val.csv
test_data=../data/test.csv
model_dir=./trained_models

for seed in {1..3}
do
    echo "Start training on seed ${seed}......"
    python train.py -s ${seed} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
###################################################################################################################
config=../config/config-bertweet.txt
train_data=../data/train.csv
dev_data=../data/val.csv
test_data=../data/test.csv
model_dir=./trained_models

for seed in {1..3}
do
    echo "Start training on seed ${seed}......"
    python train.py -s ${seed} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -mod_dir ${model_dir} -m bertweet -mul
done
###################################################################################################################
config=../config/config-bilstm.txt
train_data=../data/train.csv
dev_data=../data/val.csv
test_data=../data/test.csv
model_dir=./trained_models

for seed in {1..3}
do
    echo "Start training on seed ${seed}......"
    python train.py -s ${seed} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -mod_dir ${model_dir} -m bilstm
done
# ###################################################################################################################
config=../config/config-bilstm.txt
train_data=../data/train.csv
dev_data=../data/val.csv
test_data=../data/test.csv
model_dir=./trained_models

for seed in {1..3}
do
    echo "Start training on seed ${seed}......"
    python train.py -s ${seed} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -mod_dir ${model_dir} -m bilstm -mul
done
###################################################################################################################
