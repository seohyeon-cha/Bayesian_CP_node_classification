proj_name=cora_real
display=temp1e-3_b1
dataset=cora
nepochs=2000
wup=1e-2
id=1
runs=20
freq_path=checkpoints/cp_cora_b1_0711/frequentist
path=checkpoints/cp_cora_b1_0711/temp1e-2
n_block=1
# python main_freq.py --dataset $dataset --num_epochs $nepochs --seed 41 --gpu_id $id --training
# python main_freq.py --dataset $dataset --num_epochs $nepochs --seed 41 --gpu_id $id --wandb_save --wandb_proj_name $proj_name --display_name $display --training

# python main_freq.py --dataset $dataset --num_epochs $nepochs --seed 41 --gpu_id $id --do_cp --alpha 0.1 --num_cal 500 --num_test 1000 --path $freq_path
seed_list=($(seq 201 1 300))
for index in $(seq 0 $((${#seed_list[@]} - 1)))
do
    seed=${seed_list[$index]}
    python main_freq.py --dataset $dataset --num_epochs $nepochs --seed $seed --gpu_id $id --do_cp --alpha 0.1 --num_cal 500 --num_test 1000 --path $freq_path
done
# wait 

# python main_bbgdc.py --nblock $n_block --gpu_id $id --wup $wup  --dataset $dataset --num_epochs $nepochs --num_run $runs --training --wandb_save --wandb_proj_name $proj_name --display_name $display --seed 41 
# python main_bbgdc.py --nblock $n_block --gpu_id 0 --wup $wup --seed 41 --training  --dataset $dataset --num_epochs $nepochs --num_run $runs 

# python main_bbgdc.py --nblock $n_block --seed 41 --gpu_id 0 --wup $wup --do_cp --alpha 0.1 --num_cal 500 --num_test 1000 --path $path  --dataset $dataset --num_epochs $nepochs --num_run $runs 

# seed_list=($(seq 201 1 300))
# for index in $(seq 0 $((${#seed_list[@]} - 1)))
# do
#     seed=${seed_list[$index]}
#     python main_bbgdc.py --nblock $n_block --seed $seed --gpu_id $id --wup $wup --do_cp --alpha 0.1 --num_cal 500 --num_test 1000 --path $path  --dataset $dataset --num_epochs $nepochs --num_run $runs 
# done
# wait 