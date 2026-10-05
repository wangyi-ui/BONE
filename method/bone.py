import torch
from torch import nn
from utils.cl_loss import con_inter_view, con_reg_loss


class BONE(nn.Module):
    def __init__(self, m_list, k, a, v, layer=5, seed=1, device='cpu', anchor_init=None,
                 lamb=1e-1, temperature=1, reg=1, info=True):
        super().__init__()
        self.seed = 1
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.m_list = m_list
        self.k, self.a, self.v = k, a, v
        self.r = int(a * k)
        self.layer = layer
        self.device = device
        self.anchor_list = nn.ParameterList()
        self.fc_Z = nn.ParameterList()
        self.temperature = temperature
        self.lamb = lamb
        self.con_metric = 'cosine'
        self.reg = reg

        if anchor_init is not None and isinstance(anchor_init, list) and len(anchor_init) == v:
            init_rand = False
            if info:
                print("Init by pre-computed anchors")
        else:
            init_rand = True
            if info:
                print("Random init anchors")

        for vi in range(v):
            if init_rand:
                init_tensor = torch.randn(m_list[vi], self.r, dtype=torch.float32, device=device)
            else:
                if anchor_init[vi].shape != (m_list[vi], self.r):
                    raise ValueError(f"Wrong anchor size: {anchor_init[vi].shape}")
                init_tensor = torch.tensor(anchor_init[vi], dtype=torch.float32, device=device)
            self.anchor_list.append(nn.Parameter(init_tensor))
            fc_Z_k = nn.ParameterList()
            for k in range(layer):
                fc_Z_k.append(nn.Linear(self.r, self.r, bias=True, dtype=torch.float32, device=device))
                fc_Z_k[-1].weight.data = torch.eye(self.r, dtype=torch.float32, device=device)
            self.fc_Z.append(fc_Z_k)

        self.label_fc = nn.Sequential(
            nn.Linear(self.r, self.k),
            nn.ELU(),
            nn.Softmax(dim=1),
        )

    def forward(self, batch, test=False):
        A, Z = [], [None] * self.v
        for vi in range(self.v):
            A.append(self.anchor_list[vi])
        ret_res = {
            'loss': None,
            'label': None,
        }

        for k in range(self.layer):
            for vi in range(self.v):
                Z[vi] = self.fc_Z[vi][k](torch.linalg.inv(A[vi].T.mm(A[vi]) + self.lamb * torch.eye(
                    self.r, device=self.device)).mm(A[vi].T).mm(batch[vi].T).T).T
                A[vi] = batch[vi].T.mm(Z[vi].T).mm(torch.linalg.inv(
                    Z[vi].mm(Z[vi].T) + self.lamb * torch.eye(self.r, device=self.device)))

        Z_label, Z_embedding = [None] * self.v, [None] * self.v

        for vi in range(self.v):
            Z_label[vi] = self.label_fc(Z[vi].T).T

        weighted_z_label = [z for z in Z_label]
        z_label_stack = torch.stack(weighted_z_label).sum(dim=0) / self.v
        z_label = torch.argmax(z_label_stack, dim=0)
        z_label_v = []
        for vi in range(self.v):
            z_label_v.append(torch.argmax(Z_label[vi], dim=0))

        loss_val = torch.tensor(0.0, device=self.device)
        if not test:
            loss_val = (con_inter_view([z.T for z in Z_label], metric=self.con_metric, device=self.device) +
                          self.reg * con_reg_loss(Z_label))
        ret_res['loss'] = loss_val
        ret_res['label'] = z_label
        return ret_res
