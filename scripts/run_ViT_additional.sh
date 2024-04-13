# =================Entropy=====================
selection="Uncertainty"
per_epoch="False"
use_hard_examples="False"
uncertainty="Entropy"

fraction=0.05

seed=1
gpu=2
id=45
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${use_hard_examples}${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=3
id=46
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=0
id=47
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out


fraction=0.1

seed=1
gpu=1
id=48
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=2
id=49
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=3
id=50
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=0
id=51
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=1
id=52
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=2
id=53
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --n-class 10 --model "ViT" --selection $selection --uncertainty $uncertainty --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${use_hard_examples}_${selection}_${id}_${fraction}_${seed}.out

# =================GraphCut=====================
selection="Uniform"
per_epoch="False"
balance="True"
selection="Submodular"
submodular="GraphCut"

fraction=0.05

seed=1
gpu=1
id=54
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=2
id=55
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=3
id=56
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.1

seed=1
gpu=0
id=57
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=1
id=58
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=2
id=59
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out


fraction=0.3

seed=1
gpu=3
id=60
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=2
gpu=0
id=61
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out

seed=3
gpu=1
id=62
python src/DeepCore/main.py --gpu $gpu --lr 0.001 --optimizer "Adam"  --data_path "data/" --dataset "CIFAR10" --seed $seed --balance $balance --n-class 10 --model "ViT" --selection $selection --submodular $submodular --epochs 200 --resolution 32 --batch-size 128 --fraction $fraction --per_epoch $per_epoch >& ViT_${selection}_${id}_${fraction}_${seed}.out
