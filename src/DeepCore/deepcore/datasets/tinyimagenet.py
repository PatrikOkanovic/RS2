from torchvision import datasets, transforms
import os
import requests
import zipfile
import torch


def TinyImageNet(args, downsize=False):
    data_path = args.data_path
    if not os.path.exists(os.path.join(data_path, "tiny-imagenet-200")):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # 248MB
        print("Downloading Tiny-ImageNet")
        r = requests.get(url, stream=True)
        with open(os.path.join(data_path, "tiny-imagenet-200.zip"), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print("Unziping Tiny-ImageNet")
        with zipfile.ZipFile(os.path.join(data_path, "tiny-imagenet-200.zip")) as zf:
            zf.extractall(path=data_path)

    channel = 3
    im_size = (32, 32) if downsize else (64, 64)
    num_classes = 200
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if downsize:
        transform = transforms.Compose([transforms.Resize(32), transform])

    dst_train = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/val'), transform=transform)

    class_names = dst_train.classes
    dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
