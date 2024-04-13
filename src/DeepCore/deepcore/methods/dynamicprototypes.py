import numpy as np
from .coresetmethod import CoresetMethod
from scipy.spatial import distance
import torch
from collections import defaultdict
from scipy.special import softmax
from tqdm import tqdm


class DynamicSupervisedPrototypes(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        self.distance_type = args.embedding_distance
        self.use_hard_examples = args.use_hard_examples
        self.probabilities = None
        self.softmax = torch.nn.Softmax(dim=0)
        self.all_index_mapper = [[] for _ in range(self.num_classes)]
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
        for image, ground_truth, embedding, index in tqdm(zip(images, labels, embeddings, indices)):
            embeddings_by_label[ground_truth.item()].append((image, ground_truth, embedding, index))
            self.all_index_mapper[ground_truth.item()].append(index)

        return embeddings_by_label

    def get_hard_vs_easy(self):
        """Closest to centre will be with the least probability if in hard setting, else those distances
        are multiplied with -1.0 and become the most probable to sample from."""
        return -1.0

    def run_prototypes(self):
        embeddings_by_label = self.get_embeddings_by_label()

        self.probabilities = [None for _ in range(self.num_classes)] #np.zeros(shape=(self.num_classes, len(embeddings_by_label[0])))
        for label, data_array in embeddings_by_label.items():
            current_prob = []
            center = torch.mean(torch.stack([data[2] for data in data_array]), dim=0)
            for idx, (image, ground_truth, embedding, index) in enumerate(data_array):
                current_prob.append(self.embedding_distance(center.numpy(), embedding.numpy(), self.distance_type) * self.get_hard_vs_easy())
            self.probabilities[label] = softmax(torch.tensor(current_prob))

        # self.probabilities = softmax(self.probabilities, axis=1)

    def sample(self,):

        indices = []

        for current_class in range(self.num_classes):
            N = round(len(self.all_index_mapper[current_class]) * self.fraction)

            probabilities = self.probabilities[current_class].numpy().copy()
            # Sample without replacement
            for i in range(N):
                # Get the index of the remaining categories
                remaining_categories = np.where(probabilities > 0)[0]
                # Calculate the probabilities of the remaining categories
                remaining_probabilities = probabilities[remaining_categories] / np.sum(probabilities[remaining_categories])
                # Sample a category from the remaining categories with probabilities proportional to their probabilities
                chosen_category = np.random.choice(remaining_categories, p=remaining_probabilities)
                index = self.all_index_mapper[current_class][chosen_category]
                # Update the samples and probabilities
                indices.append(index)
                probabilities[chosen_category] = 0

        return indices

    def select(self, **kwargs):
        if self.probabilities is None:
            self.run_prototypes()
        return {"indices": self.sample()}, 0
