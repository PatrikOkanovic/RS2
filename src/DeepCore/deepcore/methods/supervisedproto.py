import numpy as np
from .coresetmethod import CoresetMethod
import torch
from collections import defaultdict
from scipy.spatial import distance
import torch
from collections import defaultdict
from tqdm import tqdm


class SupervisedPrototypes(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        self.distance_type = args.embedding_distance
        self.use_hard_examples = args.use_hard_examples
        np.random.seed(self.random_seed)

    def embedding_distance(self, x, y, distance_type):
        if distance_type == "euclidean_distance":
            return np.linalg.norm(x - y)
        elif distance_type == "cosine_similarity":
            return distance.cosine(x, y)
        else:
            raise ValueError(f"{distance_type} not supported.")

    def get_embeddings(self):
        embeddings = []
        all_images = []
        all_labels = []

        # best pretrained model: https://github.com/facebookresearch/swav
        swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')

        # counter = 3
        data_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=False, batch_size=64)
        with torch.no_grad():
            for data in tqdm(data_loader):
                images, labels = data[0], data[1]
                embeddings.append(swav(images))
                all_images.append(images)
                all_labels.append(labels)
        # print('Generated embeddings')
        return torch.cat(all_images), torch.cat(all_labels), torch.cat(embeddings)

    def get_embeddings_by_label(self,):
        # get embeddings
        images, labels, embeddings = self.get_embeddings()
        embeddings_by_label = defaultdict(list)
        indices = range(len(images))
        for image, ground_truth, embedding, index in zip(images, labels, embeddings, indices):
            embeddings_by_label[ground_truth.item()].append((image, ground_truth, embedding, index))

        return embeddings_by_label

    def run_prototypes(self):
        embeddings_by_label = self.get_embeddings_by_label()

        selected_indices = []
        for label, data_array in embeddings_by_label.items():
            center = torch.mean(torch.stack([data[2] for data in data_array]), dim=0)
            data_array.sort(key=lambda element: self.embedding_distance(center.numpy(), element[2].numpy(), self.distance_type),
                            reverse=self.use_hard_examples)
            end_index = round(len(data_array) * self.fraction)
            # data_array[label] = data_array[start_index:]
            embeddings_by_label[label] = data_array[:end_index]
            for image, ground_truth, embedding, index in embeddings_by_label[label]:
                selected_indices.append(index)

        return selected_indices


    def select(self, **kwargs):
        return {"indices": self.run_prototypes()}, 0
