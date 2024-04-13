# Description: This script is for SLURM job submissions for timeToAcc experiments on ImageNet.
# sh run_timeToAcc_imagenet_slurm.sh

data_path="data/" 
out_path="outputs/timeToAcc-ImageNet"
model="ResNet18"
epochs=200
batch_size=256
resolution=224
test_interval=1
use_hard_examples="False"
chpt_path=""

dataset="ImageNet"
n_class=1000
workers=16 # num workers for pytorch DataLoader

per_epoch="False"
balance="False"

for seed in 1 2 3
do
        for selection in "Uniform" "Forgetting" "GraNd"
        do
                for fraction in 0.01 0.05 0.1
                do
                        python src/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --workers $workers --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
​
        done
done

# herding needs balance
balance="True"
for seed in 1 2 3
do
        for selection in "Herding"
        do
                for fraction in 0.01 0.05 0.1
                do
                        python src/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --workers $workers --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
​
        done
done
# run uncertainty based methods
per_epoch="False"
balance="False"
selection="Uncertainty"
for seed in 1 2 3
do
        for uncertainty in "LeastConfidence" "Entropy" "Margin"
        do
                for fraction in 0.01 0.05 0.1
                do
                        python src/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --uncertainty $uncertainty --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --workers $workers --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}${uncertainty}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
​
        done
done
​
# run submodular
balance="True"
selection="Submodular"
for seed in 1 2 3
do
        for submodular in "FacilityLocation" "GraphCut"
        do
                for fraction in 0.01 0.05 0.1
                do
                        python src/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --submodular $submodular --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --workers $workers --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}${submodular}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
​
        done
done


# RS2 w/ replacement and w/replacement stratified
per_epoch="True"
selection="Uniform"
for seed in 1 2 3
do
        for balance in "True" "False"
        do
                for fraction in 0.01 0.05 0.1
                do
                        python src/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --workers $workers --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done

        done
done

# RS2 w/o replacement
per_epoch="True"
balance="False"
for seed in 1 2 3
do
        for selection in  "UniformNoReplacement"
        do
                for fraction in  0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                        python main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/rs2_perepoch${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done

        done
done

# run FULL methods for 2,10,20 epochs which is like 1,5,10% fraction for 200 epochs of rs2 w/o repl
# per_epoch="False"
# selection="Full"
# fraction=1
# for seed in 1 2 3
#     do
#     for epochs in 2 10 20
#         do
#             python src/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --save_path $chpt_path >& $out_path/timeToAcc_${dataset}_${model}_${selection}_${epochs}_${fraction}_${seed}.out
#     done
# done
