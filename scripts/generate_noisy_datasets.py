from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import torch
import os


def generate_dataset(dst_train, num_classes, PERCENTAGE_OF_NOISE, name):
    np.random.seed(0)
    indices_to_change = np.random.choice(range(len(dst_train)), int(PERCENTAGE_OF_NOISE * len(dst_train)))
    images = []
    labels = []
    for idx, data in tqdm(enumerate(dst_train)):
        image = data[0]
        label = data[1]

        images.append(image)
        if idx in indices_to_change:
            possible_labels = list(range(num_classes))
            possible_labels.remove(label)
            label = np.random.choice(possible_labels, 1).item()
        labels.append(label)

    images = torch.stack(images)
    tensors = [images, labels]
    torch.save({"tensors": tensors}, f"data/{name}_{int(PERCENTAGE_OF_NOISE*100)}.pt")


if __name__ == "__main__":
    data_path = "data/"
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    transform = transforms.Compose([transforms.ToTensor(), ])
    dst_train = datasets.CIFAR10(os.path.join(data_path, 'cifar10'), train=True, download=True, transform=transform)

    for percentage in [0.1, 0.3, 0.5]:
        generate_dataset(dst_train=dst_train, num_classes=num_classes, PERCENTAGE_OF_NOISE=percentage,
                         name="noisycifar10")
