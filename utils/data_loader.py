import torch
from torch.utils.data import Dataset


class RandDataset(Dataset):
    def __init__(self, data, label, device='cpu'):
        """
        data: ndarray (n * m),
        label_list: list (n),
        """
        self.data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
        self.label = torch.from_numpy(label).to(device=device, dtype=torch.float32)

    def __getitem__(self, item):
        return self.data[item, :], self.label[item]

    def __len__(self):
        return self.data.shape[0]


class RandDatasetMV(Dataset):
    def __init__(self, data_list, label_list, device='cpu'):
        """
        data: list of ndarray v * (m * n),
        label_list: list (n),
        """
        self.view = len(data_list)
        self.data_list, self.label_list = [], []
        for i in range(self.view):
            self.data_list.append(torch.from_numpy(data_list[i]).to(device=device, dtype=torch.float32))
        self.label_list = torch.from_numpy(label_list).to(device=device, dtype=torch.float32)
        self.device = device

    def __getitem__(self, item):
        return [data.T[item, :] for data in self.data_list], self.label_list[item]

    def __len__(self):
        return len(self.label_list)

    def data_shape(self):
        return f'{len(self.data_list)} * {self.data_list[-1].shape}'


class RandDatasetRL(Dataset):
    def __init__(self, data, embedding, label, device='cpu'):
        self.data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
        self.embedding = torch.from_numpy(embedding).to(device=device, dtype=torch.float32)
        self.label = torch.from_numpy(label).to(device=device, dtype=torch.float32)

    def __getitem__(self, item):
        return self.data[item, :], self.embedding[item, :], self.label[item]

    def __len__(self):
        return self.data.shape[0]
