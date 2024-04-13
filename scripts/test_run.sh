# Description: This script is for SLURM job submissions for timeToAcc experiments on CIFAR10
# sh run_timeToAcc_cifar10_slurm.sh

data_path="data/" # path to data folder containing CIFAR10, CIFAR100, ImageNet30, ImageNet, TinyImageNet, NoisyCIFAR10
out_path="outputs/timeToAcc-CIFAR10" # path to save outputs
model="ResNet18"
mkdir -p $out_path
epochs=200
batch_size=128 # 128 for cifar10, 256 for imagenet
resolution=32 # 32 for cifar10, 224 for imagenet
test_interval=1 # epoch test interval
use_hard_examples="False"
chpt_path=""
dataset="CIFAR10"
n_class=10 # 10 for cifar10, 100 for cifar100, 1000 for imagenet, 30 for imagenet30, 200 for tinyimagenet

# RS2 w/o replacement
per_epoch="True"
balance="False"
for seed in 1
do
        for selection in  "UniformNoReplacement"
        do
                for fraction in  0.1
                do
                        python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/rs2_perepoch${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done

        done
done