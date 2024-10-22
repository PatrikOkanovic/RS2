from torchvision import datasets, transforms
from torchvision import transforms as T
from torch import tensor, long

def CIFAR100(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR100(args.data_path+'/cifar100', train=True, download=True, transform=train_transform)
    dst_test = datasets.CIFAR100(args.data_path+'/cifar100', train=False, download=True, transform=test_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test