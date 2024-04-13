data_path="data/" # path to data folder containing CIFAR10, CIFAR100, ImageNet30, ImageNet, TinyImageNet, NoisyCIFAR10
out_path="outputs/data_distillation" # path to save outputs
model="ConvNet"
epochs=1000
balance="True" # we only test stratified version
per_epoch="True"
lr=0.01
scheduler="CosineAnnealingLR"
test_interval=1

# ======================CIFAR10====================
dataset="CIFAR10"
n_class=10

batch_size=10
fraction=0.0002
for selection in "Uniform" 
do
  for seed in 1 2 3 
  do
  	sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done

batch_size=100
fraction=0.002
for selection in "Uniform" 
do
  for seed in 1 2 3 
  do
  	sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch  --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done

batch_size=500
fraction=0.01
for selection in "Uniform" 
do
  for seed in 1 2 3 
  do
  	sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch  --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done


# ======================CIFAR100====================

dataset="CIFAR100"
n_class=100

batch_size=100
fraction=0.002
for selection in "Uniform"
do
  for seed in 1 2 3 
  do
  	sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done

batch_size=1000
fraction=0.02
for selection in "Uniform"
do
  for seed in 1 2 3 
  do
  	sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch  --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done

batch_size=5000
fraction=0.1
for selection in "Uniform"
do
  for seed in 1 2 3 
  do
  	sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch  --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done


#=========================TinyImageNet====================
dataset="TinyImageNet"
n_class=200

batch_size=200
fraction=0.002
for selection in "Uniform" 
do
  for seed in 1 2 3 
  do
        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done

batch_size=200
fraction=0.02
for selection in "Uniform" 
do
  for seed in 1 2 3 
  do
        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch  --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done

batch_size=300
fraction=0.1
for selection in "Uniform" 
do
  for seed in 1 2 3 
  do
        sbatch -n 1 --cpus-per-task=1 --gpus=rtx_3090:1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python src/DeepCore/main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --selection $selection --epochs $epochs --batch-size $batch_size --balance $balance --fraction $fraction --seed $seed --per_epoch $per_epoch  --lr $lr --scheduler $scheduler --test_interval $test_interval >& distillation_${dataset}_${model}_${selection}_${balance}_${fraction}_${seed}.out"
  done
done
