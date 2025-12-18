#!/bin/bash
# graph datasets: 'wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'

# example: ./run_link_prediction.sh uci 1 0 DyGFormer
# explain: ./run_link_prediction.sh $dataset_name $gpu $par_id $model_name

echo 

# 检查是否存在动态参数
if [ "$#" -ge 1 ]; then
    dataset_name="$1"
else
    dataset_name="wikipedia"
fi

if [ "$#" -ge 2 ]; then
    gpu="$2"
else
    gpu=0
fi


if [ "$#" -ge 4 ]; then
    par_id="$4"
else
    par_id=0
fi

if [ "$#" -ge 5 ]; then
    model_name="$5" # DyGMamba
else
    model_name="DyGMamba"
fi


comments="default"


if [ "$par_id" -ne 0 ]; then
    comments="${comments}_${par_id}"
fi


start_time=$(date +%s)

outfile=${dataset_name}_${comments}_${model_name}.out

# 记录脚本开始运行的时间
current_time=$(date)
echo "Current time is: $current_time" > $outfile


# count of gpu num
gpu_count=$(nvidia-smi -L | wc -l)

host_name=$(hostname)
echo "Running $model_name model for data $dataset_name on $host_name:($gpu), setting: $comments" >> $outfile


nohup python train_link_prediction.py --dataset_name ${dataset_name} --model_name ${model_name} --load_best_configs --gpu ${gpu} \
                                      --comments ${comments} >> $outfile &
pid1=$!
# wait $pid1

