#!/bin/bash


model="GAT"
dataset="citeseer"
split="public"
random_seed='42'
device='1'
weight_decay='5e-4'


num_layers=(2 3 4)
hid_sizes=(16 512)
drop_outs=(0.2 0.6)
lrs=(0.01 0.001)
num_runs=3
epochs=400


log_dir="logs"
mkdir -p $log_dir


log_file="./${log_dir}/${model}_search_${dataset}.log"
echo "Hyperparameter Search Log for dataset: $dataset" > $log_file

for num_layer in "${num_layers[@]}"; do
    for hid_size in "${hid_sizes[@]}"; do
        for drop_out in "${drop_outs[@]}"; do
            for lr in "${lrs[@]}"; do
                cmd="/home/user/anaconda3/envs/wcp-torch-2.4.0/bin/python /home/user/WCP/workspace/PythonProject/GraphTraining/node_classification.py \
                    --dataset $dataset \
                    --model $model \
                    --self_loop \
                    --random_seed $random_seed \
                    --split $split \
                    --num_layers $num_layer \
                    --hid_size $hid_size \
                    --drop_out $drop_out \
                    --gat_atten_drop 0.6 \
                    --lr $lr \
                    --weight_decay $weight_decay \
                    --num_runs $num_runs \
                    --epochs $epochs \
                    --device $device"
                echo "$cmd" >> $log_file
                result=$($cmd)
                echo "$result" >> $log_file
            done
        done
    done
done


echo "Hyperparameter search completed. Results saved to $log_file."
