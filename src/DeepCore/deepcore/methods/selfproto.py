import numpy as np
from .coresetmethod import CoresetMethod
from scipy.spatial import distance
import torch
from collections import defaultdict
from sklearn.cluster import KMeans


class SelfSupervisedPrototypes(CoresetMethod):
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

    def run_clustering(self):
        # np.random.seed(self.random_seed)
        images, labels, embeddings = self.get_embeddings()
        # cluster the embeddings
        kmeans = KMeans(init="k-means++", n_clusters=self.num_classes, n_init=4, random_state=0)
        kmeans.fit(embeddings)
        kmeans_clustered_points = defaultdict(list)
        # prepare for sorting the data
        indices = range(len(images))
        for image, ground_truth, embedding, kmeans_label, index in zip(images, labels, embeddings, kmeans.labels_, indices):

            kmeans_clustered_points[kmeans_label].append((image, ground_truth,
                                                          self.embedding_distance(kmeans.cluster_centers_[kmeans_label],
                                                                             embedding.numpy(), self.distance_type),
                                                          index))

        # sort by distance from cluster center and get subset
        selected_indices = []
        for kmeans_label, array in kmeans_clustered_points.items():
            array.sort(key=lambda element: element[2], reverse=self.use_hard_examples)
            end_index = round(len(array) * self.fraction)
            kmeans_clustered_points[kmeans_label] = array[:end_index]
            for image, ground_truth, embed_distance, index in kmeans_clustered_points[kmeans_label]:
                selected_indices.append(index)

        return selected_indices

    def get_embeddings(self):
        embeddings = []
        all_images = []
        all_labels = []

        # best pretrained model: https://github.com/facebookresearch/swav
        swav = torch.hub.load('facebookresearch/swav:main', 'resnet50', pretrained=True, force_reload=True)

        # counter = 3
        data_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=False, batch_size=64)
        with torch.no_grad():
            for data in data_loader:
                images, labels = data[0], data[1]
                embeddings.append(swav(images))
                all_images.append(images)
                all_labels.append(labels)
        # print('Generated embeddings')
        return torch.cat(all_images), torch.cat(all_labels), torch.cat(embeddings)

    def select(self, **kwargs):
        return {"indices": self.run_clustering()}, 0
