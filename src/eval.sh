# bash ./eval.sh > eval.log

###################################################################################################################
config=../config/config-bertweet.txt
test_data=../data/BiLSTM_target_classification.csv
model_dir=./trained_models

for seed in {1..3}
do
    echo "Start training on seed ${seed}......"
    python eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
###################################################################################################################
config=../config/config-bertweet.txt
test_data=../data/KPTimes_target_generation.csv
model_dir=./trained_models

for seed in {1..3}
do
    echo "Start training on seed ${seed}......"
    python eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
###################################################################################################################
config=../config/config-bertweet.txt
test_data=../data/KPTimes_target_generation_zero_shot.csv
model_dir=./trained_models

for seed in {1..3}
do
    echo "Start training on seed ${seed}......"
    python eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
###################################################################################################################