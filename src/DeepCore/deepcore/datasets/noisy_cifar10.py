import torch
from torchvision import datasets, transforms
from torchvision import transforms as T
from torch import tensor, long
import os


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), don't need this check
        self.tensors = tensors
        self.transform = transform
        self.targets = tensors[1]

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        # return self.tensors[0].size(0)
        return len(self.tensors[0])


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
    dst_test = datasets.CIFAR10(args.data_path+'/cifar10', train=False, download=True, transform=test_transform)
    class_names = dst_test.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test