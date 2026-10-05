from time import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from method.amv import amv
from method.bone import BONE
from utils.data_loader import RandDatasetMV
from utils.data_util import get_dataset, D_YouTubeFace50
from utils.method_util import get_cluster_metrics

cuda = False
device = torch.device("cuda:0") if cuda and torch.cuda.is_available() else torch.device("cpu")
data_name = D_YouTubeFace50
batch_size, n_anchor, layer = 512, 1, 2  # ACC: 0.7001, NMI: 0.8426, Purity: 0.7544
print(f"Batch: {batch_size}, n_anchor: {n_anchor}, layer: {layer}")
global_seed = 44
n_epoch = 100
dataset = get_dataset(data_name=data_name)
dataset.print_info()
dataset.shuffle()
dataset.standard_data()

start = time()
a_init = amv(dataset, m=n_anchor, ret_anchor=True, max_iter=100, info=True,
             info_iter=False, sample=0.01, seed=global_seed)
init_time = time() - start
dataset_tensor = dataset.get_tensor(device=device)
train_loader = DataLoader(dataset=RandDatasetMV(dataset.data, dataset.label, device=device), batch_size=batch_size,
                          generator=torch.Generator().manual_seed(global_seed),
                          num_workers=0, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=RandDatasetMV(dataset.data, dataset.label, device=device), batch_size=10 * batch_size,
                         generator=torch.Generator().manual_seed(global_seed),
                         num_workers=0, shuffle=False, drop_last=False)
model = BONE(m_list=dataset.m_list, k=dataset.k, a=n_anchor, v=dataset.v,
             anchor_init=a_init, layer=layer, lamb=1e-1, device=device)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")
optimizer = optim.Adam(model.parameters(), lr=1e-2)
tic = False
start = time()

for e in range(n_epoch):
    model.train()

    for j, batch_i in enumerate(train_loader):
        optimizer.zero_grad()
        data_i, label_i = batch_i
        res = model(data_i)
        loss_j = res['loss']
        loss_j.backward()
        optimizer.step()
        if j >= 10:
            break

    if not tic:
        if (e + 1) % 20 == 0:
            model.eval()
            label_list, label_i_list = [], []
            for j, batch_i in enumerate(test_loader):
                data_i, label_i = batch_i
                res = model(data_i, test=True)
                label_list.extend(res['label'].cpu().numpy())
            acc, nmi, purity = get_cluster_metrics(dataset.label, np.array(label_list), nmi=False)
            print(f"Epoch {e + 1}, ACC: {acc:.4f}, NMI: {nmi:.4f}, Purity: {purity:.4f}")
if tic:
    train_time = time() - start
    print(f"Total time cost: {train_time + init_time:.3f}s ({init_time:.3f}s, {train_time:.3f}s)")
