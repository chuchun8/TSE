# Reproducing Table 3


for seed in 0 112 342; do
    for model in "Bert" "Bertweet"; do
        CUDA_VISIBLE_DEVICES=1 python train_model.py  --dataset=Stance_Merge_Unrelated --model_select=$model --seed=$seed
    done
done

# BiLSTM
for seed in 0 112 342; do
    CUDA_VISIBLE_DEVICES=1 python train_model.py --dataset=Stance_Merge_Unrelated --model_select=BiLSTM --lr=5e-3 --seed=$seed
done

# Get results for individual datasets
python utils/eval_utils_outputs.py