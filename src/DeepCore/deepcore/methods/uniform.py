import numpy as np
from .coresetmethod import CoresetMethod


class Uniform(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        self.all_index = np.arange(self.n_train)
        self.extremely_small_data = args.extremely_small_data
        np.random.seed(self.random_seed)

    def select_balance(self):
        """The same sampling proportions were used in each class separately."""
        # np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        current_index = []
        if not self.extremely_small_data:
            # current_index = []
            for c in range(self.num_classes):
                c_index = (self.dst_train.targets == c)
                # self.index = np.append(self.index,
                #                        np.random.choice(self.all_index[c_index], round(self.fraction * c_index.sum().item()),
                #                                         replace=self.replace))
                current_index.extend(np.random.choice(self.all_index[c_index], round(self.fraction * c_index.sum().item()),
                                                      replace=self.replace))
            # self.index = np.array(current_index).ravel().astype(np.int64)
        else:
            # first add an example for each class
            for c in range(self.num_classes):
                c_index = (self.dst_train.targets == c)
                # self.index = np.append(self.index,
                #                        np.random.choice(self.all_index[c_index], 1,
                #                                         replace=self.replace))
                current_index.extend(np.random.choice(self.all_index[c_index], 1, replace=self.replace))
            # choose randomly the rest of classes
            images_left = round(self. fraction * self.n_train) - self.num_classes
            current_replace = images_left > self.num_classes
            randomly_chosen_classes = np.random.choice(range(self.num_classes), images_left, replace=current_replace)
            for c in randomly_chosen_classes:
                c_index = (self.dst_train.targets == c)
                # self.index = np.append(self.index,
                #                        np.random.choice(self.all_index[c_index], 1,
                #                                         replace=self.replace))
                current_index.extend(np.random.choice(self.all_index[c_index], 1, replace=self.replace))
        self.index = np.append(self.index, current_index)
        return self.index

    def select_no_balance(self):
        # np.random.seed(self.random_seed)
        self.index = np.random.choice(self.all_index, round(self.n_train * self.fraction),
                                      replace=self.replace)

        return self.index

    def select(self, **kwargs):
        return {"indices": self.select_balance() if self.balance else self.select_no_balance()}, 0
