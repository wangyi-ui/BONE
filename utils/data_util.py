import gzip
import os
import numpy as np
import scipy
import scipy.io as scio
from utils.MVDataset import MVDataset
from io import BytesIO

D_MNIST_USPS = "MNIST-USPS"
D_BDGP = "BDGP"
D_CCV = "CCV"
D_Caltech5V = "Caltech-5V"
D_HandWrite = "HandWrite"
D_COIL20 = "COIL20"
D_MSRC = "MSRC"
D_YouTubeFace50 = "YouTubeFace50"
D_ALOI = "ALOI"
PATH = os.path.join(os.getcwd(), 'data')

dataset_info = {
    # data_name: [n, view, classes, (m of views)]
    D_MNIST_USPS: [5000, 2, 10, (784, 784)],
    D_BDGP: [2500, 2, 5, (1750, 79)],
    D_CCV: [6773, 3, 20, (5000, 5000, 4000)],
    D_Caltech5V: [1400, 5, 7, (40, 254, 928, 512, 1984)],
    D_HandWrite: [2000, 6, 10, (216, 76, 64, 6, 240, 47)],
    D_COIL20: [1440, 3, 20, (1024, 3304, 6750)],
    D_MSRC: [210, 4, 7, (24, 512, 256, 254)],
    D_YouTubeFace50: [126054, 4, 50, (944, 576, 512, 640)],
    D_ALOI: [10800, 4, 108, (77, 13, 64, 125)],
}


def get_dataset(data_name, path=PATH):
    """
    Load and process a multi-view dataset from a .mat file.

    The dataset is expected to contain a label vector 'Y' (1 x n) and data matrices 'Xi' (mi x n),
    where each 'Xi' corresponds to the i-th view. Returns a structured dataset object with
    views and labels.
    """
    mat_name, n_view = data_name, None
    mat_path = os.path.join(path, f'{mat_name}.mat')
    mat = scipy.io.loadmat(mat_path)
    data_list = []
    label_list = mat['Y'].astype(np.int32).reshape(-1)
    if n_view is None:
        n_view = len(dataset_info[mat_name][-1])
    for i in range(n_view):
        view_str = f"X{i + 1}"
        data_view = mat[view_str].astype(np.float32)
        if data_view.shape[0] == len(label_list):
            data_view = data_view.T
        data_view = data_view.reshape(data_view.shape[0], -1)
        data_list.append(data_view)
    return MVDataset(data=data_list, label=label_list)


def save_dataset(mv_data, name, path=PATH):
    data = mv_data.data
    label = mv_data.label
    data_dict = {'Y': label}
    for i, data_i in enumerate(data):
        data_dict[f'X{i + 1}'] = data_i
    path_full = os.path.join(path, name + '.mat')
    scio.savemat(path_full, data_dict)
    print(path_full + ' already saved')
