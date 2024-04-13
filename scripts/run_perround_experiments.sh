data_path="data/" # path to data folder containing CIFAR10, CIFAR100, ImageNet30, ImageNet, TinyImageNet, NoisyCIFAR10
out_path="outputs/perroundRS-CIFAR10" # path to save outputs
model="ResNet18"
epochs=200
batch_size=128 # 128 for cifar10, 256 for imagenet
resolution=32 # 32 for cifar10, 224 for imagenet
test_interval=1 # epoch test interval
use_hard_examples="False"
chpt_path=""
dataset="CIFAR10"
n_class=10 # 10 for cifar10, 100 for cifar100, 1000 for imagenet, 30 for imagenet30, 200 for tinyimagenet
# 1) running RS experiments
# several pruning methods (--mem-per-cpu=10240)
per_epoch="True" # True for sampling per epoch, False for sampling once
balance="False" # True for balanced sampling (stratified on classes), False for unbalanced sampling
for seed in 1 2 3
do
        for selection in "ForgettingSampling" "GraNdSampling" "CalSampling" "CraigSampling"
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
        done
done



# run uncertainty based methods (--mem-per-cpu=10240)
balance="False"
selection="Uncertainty"
for seed in 1 2 3
do
        for uncertainty in "LeastConfidence" "Entropy" "Margin"
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --uncertainty $uncertainty --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}${uncertainty}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
        done
done

# run dynamic prototypes
for selection in "DynamicSupervisedPrototypes"
do
        for seed in 1 2 3
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
       done
done

# 2) Run RC experiments
out_path="outputs/perroundRC-CIFAR10" # path to save outputs

for seed in 1 2 3
do
        for selection in "ContextualDiversity"  "Craig" "kCenterGreedy" "Forgetting" "GraNd" "Cal"
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main_pe.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
        done
done

balance="False"
selection="Uncertainty"
for seed in 1 2 3
do
        for uncertainty in "LeastConfidence" "Entropy" "Margin" "Glister"
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main_pe.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --uncertainty $uncertainty --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}${uncertainty}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
        done
done

balance="True" # True for herding
for seed in 1 2 3
do
        for selection in "Herding"
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main_pe.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done
        done
done


# RS2 w/ replacement and w/replacement stratified
selection="Uniform"
for seed in 1 2 3
do
        for balance in "True" "False"
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval --save_path $chpt_path >& $out_path/timeToAcc_${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done

        done
done

# RS2 w/o replacement
balance="False"
for seed in 1 2 3
do
        for selection in  "UniformNoReplacement"
        do
                for fraction in 0.05 0.1 0.3
                do
                        python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --resolution $resolution --test_interval $test_interval >& $out_path/rs2_perepoch${per_epoch}_${resolution}_${test_interval}_${dataset}_${model}_${selection}_balance${balance}_hard${use_hard_examples}_${fraction}_${seed}.out
                done

        done
done