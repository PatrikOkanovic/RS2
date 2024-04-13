# =================RS2=====================
selection="UniformNoReplacement"
per_epoch="True"

fraction=0.2

seed=1
gpu=0
id=0
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=1
id=1
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=2
id=2
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=3
id=3
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=0
id=4
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=1
id=5
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out


fraction=0.4

seed=1
gpu=2
id=6
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=3
id=7
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=0
id=8
python src/DeepCore/main.py --gpu $gpu  --data_path "data/" --dataset "CIFAR100" --seed $seed --n-class 100 --model "ResNet50" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ResNet50_${selection}_${id}_${fraction}_${seed}.out
