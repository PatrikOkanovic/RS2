# =================SSP EASY=====================
selection="SelfSupervisedPrototypes"
per_epoch="False"
use_hard_examples="False"

fraction=0.05

seed=1
gpu=2
id=27
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${use_hard_examples}${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=3
id=28
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=0
id=29
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out


fraction=0.1

seed=1
gpu=1
id=30
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=2
id=31
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=3
id=32
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=0
id=33
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=1
id=34
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=2
id=35
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

# =================SSP HARD=====================
selection="SelfSupervisedPrototypes"
per_epoch="False"
use_hard_examples="True"

fraction=0.05

seed=1
gpu=3
id=36
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${use_hard_examples}${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=0
id=37
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=1
id=38
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out


fraction=0.1

seed=1
gpu=2
id=39
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=3
id=40
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=0
id=41
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=1
id=42
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=2
id=43
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=3
id=44
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --use_hard_examples $use_hard_examples --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out
