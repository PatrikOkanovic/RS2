# Downloading datasets

## CIFAR-10, CIFAR-100, TinyImageNet

Downloaded automatically when running the scripts using each dataset.

For example, running the following script will create the necessary folder structure and save CIFAR-10:

```eval
source run_timeToAcc_cifar10.sh
```

## ImageNet-30

For ImageNet-30, we download the dataset and extract to `data/`, from: [here](https://github.com/hendrycks/ss-ood)

## ImageNet-1k

1. Download using the following link: [here](https://www.image-net.org/download.php).
2. Extract the dataset and move to `data/`
3. Following the instructions for [`torchvision.datasets.ImageFolder`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) from this [link](https://docs.nvidia.com/deeplearning/dali/archives/dali_08_beta/dali-developer-guide/docs/examples/pytorch/resnet50/pytorch-resnet50.html), we 
organize the validation folder of ImageNet-1k by running [this](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) script in `data/imagenet1k/ILSVRC/Data/CLS-LOC/val`


## Noisy CIFAR-10
Running the following command will generate the necessary datasets for experiments on CIFAR-10 with label noise:
```bash
python generate_noisy_datasets.py
```

When the datasets are generate, one can choose the dataset with noise ratio _p%_
by running `main.py` with arguments `--dataset NoisyCIFAR10`, and `--noisy_dataset_path noisycifar10_{p}.pt`
