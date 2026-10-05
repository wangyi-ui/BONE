import torch
from torch import optim
from torch.utils.data import DataLoader
from method.amv import amv
from method.bone import BONE
from utils.data_loader import RandDatasetMV
from utils.data_util import (get_dataset, D_HandWrite, D_BDGP, D_MNIST_USPS,
                             D_CCV, D_COIL20, D_Caltech5V, D_ALOI, D_MSRC)
from utils.method_util import get_cluster_metrics
from utils.seed_util import set_seed

setting_dict = {  # BONE
    D_BDGP: (512, 2, 2),  # ACC: 0.9924, NMI: 0.9728, Purity: 0.9924
    D_HandWrite: (128, 4, 3),  # ACC: 0.9770, NMI: 0.9442, Purity: 0.9770
    D_MNIST_USPS: (512, 4, 2),  # ACC: 0.9816, NMI: 0.9515, Purity: 0.9816
    D_COIL20: (128, 1, 4),  # ACC: 0.7688, NMI: 0.8436, Purity: 0.7896
    D_Caltech5V: (128, 4, 3),  # ACC: 0.9057, NMI: 0.8306, Purity: 0.9057
    D_MSRC: (128, 3, 5),  # ACC: 0.9381, NMI: 0.8809, Purity: 0.9381
    D_ALOI: (512, 2, 2),  # ACC: 0.7836, NMI: 0.8563, Purity: 0.7991
    # D_CCV: (512, 3, 2),  # ACC: 0.3446, NMI: 0.3042, Purity: 0.3662
}

global_seed = 42
n_epoch = 100
cuda = False
device = torch.device("cuda:0") if cuda and torch.cuda.is_available() else torch.device("cpu")
set_seed(global_seed)

data_list = list(setting_dict.keys())
for data_i in data_list:
    print("-" * 50)
    batch_size, n_anchor, layer = setting_dict[data_i]
    print(f"Data: {data_i}, batch: {batch_size}, n_anchor: {n_anchor}, layer: {layer}")
    dataset = get_dataset(data_name=data_i)
    dataset.print_info()
    dataset.shuffle()
    dataset.standard_data()
    a_init = amv(dataset, m=n_anchor, ret_anchor=True, max_iter=100, info=True, info_iter=False, seed=global_seed)
    dataset_tensor = dataset.get_tensor(device=device)
    train_loader = DataLoader(dataset=RandDatasetMV(dataset.data, dataset.label, device=device), batch_size=batch_size,
                              generator=torch.Generator().manual_seed(global_seed),
                              num_workers=0, shuffle=True, drop_last=True)
    model = BONE(m_list=dataset.m_list, k=dataset.k, a=n_anchor, v=dataset.v,
                 anchor_init=a_init, layer=layer, lamb=1e-1, seed=global_seed, device=device)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for e in range(n_epoch):
        model.train()

        for j, batch_i in enumerate(train_loader):
            optimizer.zero_grad()
            data_i, label_i = batch_i
            res = model(data_i)
            loss_j = res['loss']
            loss_j.backward()
            optimizer.step()

        if (e + 1) % 20 == 0:
            model.eval()
            res = model(dataset_tensor, test=True)
            acc, nmi, purity = get_cluster_metrics(dataset.label, res['label'].cpu().numpy(), nmi=False)
            print(f"Epoch {e + 1}, ACC: {acc:.4f}, NMI: {nmi:.4f}, Purity: {purity:.4f}")
