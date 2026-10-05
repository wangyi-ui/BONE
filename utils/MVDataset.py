import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class MVDataset:
    def __init__(self, data, label):
        """
        :param data: list of (mi, n)
        :param label: list of n
        """
        self.data = data
        self.data_tensor = None
        self.label = label
        self.m_list = []
        self.n = len(label)
        self.v = len(data)
        self.k = len(set(label))
        for data_i in data:
            self.m_list.append(data_i.shape[0])
            if data_i.shape[1] != self.n:
                raise ValueError(f"Data shape error ({data_i.shape[1]} != {self.n})")

    def get_info(self):
        return self.m_list, self.n, self.v

    def print_info(self):
        print(f'm({self.m_list}), n({self.n}), v({self.v}), k({self.k})')

    def shuffle(self, seed=42):
        np.random.seed(seed)
        random_indices = np.random.permutation(self.n)
        for vi in range(self.v):
            self.data[vi] = self.data[vi][:, random_indices]
        self.label = self.label[random_indices]

    def standard_data(self):
        for view in range(self.v):
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            standardized_view = scaler.fit_transform(self.data[view].T)
            self.data[view] = standardized_view.T

    def min_max_scale_data(self):
        for view in range(self.v):
            scaler = MinMaxScaler()
            standardized_view = scaler.fit_transform(self.data[view].T)
            self.data[view] = standardized_view.T

    def get_tensor(self, device='cpu'):
        if self.data_tensor is None:
            self.data_tensor = []
            for vi in range(self.v):
                self.data_tensor.append(torch.from_numpy(self.data[vi].T).to(device=device, dtype=torch.float32))
        return self.data_tensor
