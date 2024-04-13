import numpy as np
from .coresetmethod import CoresetMethod


class UniformNoReplacement(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        self.all_index = np.arange(self.n_train)
        np.random.seed(self.random_seed)
        self.start, self.end = None, None

    def sample(self):
        if self.start < self.end:
            return self.all_index[self.start:self.end]
        else:
            return np.concatenate((self.all_index[self.start:], self.all_index[:self.end]))

    def reset(self):
        self.start = 0
        self.end = self.coreset_size
        np.random.shuffle(self.all_index)

    def select(self, **kwargs):
        if self.start is None or self.end < self.start or self.end == len(self.all_index):
            self.reset()
        else:
            self.start = self.end
            self.end = (self.end + self.coreset_size) % (len(self.all_index)+1)
        return {"indices": self.sample()}, 0
