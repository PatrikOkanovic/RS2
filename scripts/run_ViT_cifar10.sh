
# =================RS2=====================
selection="UniformNoReplacement"
per_epoch="True"

fraction=0.05

seed=1
gpu=0
id=0
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=1
id=1
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=2
id=2
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.1

seed=1
gpu=3
id=3
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=0
id=4
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=1
id=5
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=2
id=6
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=3
id=7
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=0
id=8
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


# =================Uniform=====================
selection="Uniform"
per_epoch="False"

fraction=0.05

seed=1
gpu=1
id=9
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=2
id=10
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=3
id=11
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.1

seed=1
gpu=0
id=12
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=1
id=13
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=2
id=14
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=3
id=15
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=0
id=16
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=1
id=17
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


# =================Forgetting=====================
selection="Craig"
per_epoch="False"

fraction=0.05

seed=1
gpu=2
id=18
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=3
id=19
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=0
id=20
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.1

seed=1
gpu=1
id=21
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=2
id=22
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=3
id=23
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=0
id=24
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=1
id=25
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=2
id=26
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

