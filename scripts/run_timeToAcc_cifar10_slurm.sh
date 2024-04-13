# Description: This script is for SLURM job submissions for timeToAcc experiments on CIFAR10
# source run_timeToAcc_cifar10_slurm.sh

data_path="data/" # path to data folder containing CIFAR10, CIFAR100, ImageNet30, ImageNet, TinyImageNet, NoisyCIFAR10
out_path="outputs/timeToAcc-CIFAR10" # path to save outputs
mkdir -p $out_path
model="ResNet18"
epochs=200
batch_size=128 # 128 for cifar10, 256 for imagenet
resolution=32 # 32 for cifar10, 224 for imagenet
test_interval=1 # epoch test interval
use_hard_examples="False"
dataset="CIFAR10"
n_class=10 # 10 for cifar10, 100 for cifar100, 1000 for imagenet, 30 for imagenet30, 200 for tinyimagenet

# methods:
per_epoch="False" # True for sampling per epoch, False for sampling once
balance="False" # True for balanced sampling (stratified on classes), False for unbalanced sampling
for seed in 1 2 3
do
        for selection in "Uniform" "ContextualDiversity" "kCenterGreedy" "Forgetting" "GraNd" "Cal" "Glister"
        do
                for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                       sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                done
        done
done

balance="True" # True for herding
for seed in 1 2 3
do
        for selection in "Herding"
        do
                for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                done
        done
done
balance="True" # True for herding
# cal and craig need more memory
for seed in 1 2 3
do
        for selection in "Cal" "Craig"
        do
                for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=100G --tmp=20000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                done
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
                for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --uncertainty $uncertainty --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}${uncertainty}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                done
        done
done
# run submodular
balance="True"
selection="Submodular"
for seed in 1 2 3
do
        for submodular in "FacilityLocation" "GraphCut"
        do
                for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --submodular $submodular --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}${submodular}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                done
        done
done


# run prototype methods
per_epoch="False"
for use_hard_examples in "True" "False"
do
  for selection in "SelfSupervisedPrototypes" "SupervisedPrototypes"
  do
          for seed in 1 2 3
          do
                  for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                  do
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --use_hard_examples $use_hard_examples --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                  done
          done
  done
done

# run dynamic prototypes
per_epoch="True"
for selection in "DynamicSupervisedPrototypes"
do
        for seed in 1 2 3
        do
                for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                done
       done
done

# active learning methods (AL)
n_query=1000
cycle=46

method="Uncertainty"
for seed in 1 2 3
do
  for uncertainty in "Margin" "Entropy" "LeastConfidence"
  do
    sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=24:00:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/AL/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --method $method --uncertainty $uncertainty --epochs $epochs --batch-size $batch_size --n-query $n_query --seed $seed --resolution $resolution --cycle $cycle >& $out_path/timeToAcc_${dataset}_${model}_${resolution}_${method}${uncertainty}_${seed}.out"
  done
done

# RS2 w/ replacement and w/replacement stratified
per_epoch="True"
selection="Uniform"
for seed in 1 2 3
do
        for balance in "True" "False"
        do
                for fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                do
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
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
                        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/rs2_perepoch${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out"
                done

        done
done

# FULL dataset w/ early stopping
# per_epoch="False"
# selection="Full"
# fraction=1
# for seed in 1 2 3
# do
#     for epochs in 2 10 20 40 60 80 100 120 140 160 180 200
#     do
#                 sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=3:59:00 --job-name="cifar10_time_to_acc" --mem-per-cpu=50G --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution >& $out_path/timeToAcc_${dataset}_${model}_${selection}_${epochs}_${fraction}_${seed}.out"
#     done
# done
