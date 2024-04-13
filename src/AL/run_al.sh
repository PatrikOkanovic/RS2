data_path="../data"
dataset="CIFAR100"
n_class=100
model="ResNet18"
method="Uniform"
uncertainty="Margin"
epochs=200
batch_size=128
#balance="False"
#per_epoch="False"

for seed in  1 2 3 #4 5
  do
    for fraction in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        sbatch -n 1 --cpus-per-task=1 --gpus=1 --time=23:59:00 --mem-per-cpu=10240 --tmp=10000 --wrap="python main.py --data_path $data_path --dataset $dataset --n-class $n_class --model $model --method $method --uncertainty $uncertainty --epochs $epochs --batch-size $batch_size --fraction $fraction --seed $seed >& check_${dataset}_${model}_${method}_${uncertainty}_${balance}_${fraction}_${seed}.out"
    done
  done

