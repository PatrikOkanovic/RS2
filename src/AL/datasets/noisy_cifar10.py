import torch
from torchvision import datasets, transforms
from torchvision import transforms as T
from torch import tensor, long
from utils import CustomTensorDataset
import os


def NoisyCIFAR10(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    train_transform = T.Compose([T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    tensors = torch.load(os.path.join(args.data_path, args.noisy_dataset_path))["tensors"]
    dst_train = CustomTensorDataset(tensors=tensors, transform=train_transform)
    dst_unlabeled = CustomTensorDataset(tensors=tensors, transform=train_transform)
    dst_test = datasets.CIFAR10(args.data_path+'/cifar10', train=False, download=True, transform=test_transform)
    class_names = dst_test.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test