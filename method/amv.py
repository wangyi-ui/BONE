from time import time
import numpy as np


def amv(mv_data, m=1, max_iter=100, seed=42, ret_anchor=False, sample=1, info=True, info_iter=True):
    v = mv_data.v
    k = mv_data.k
    n = mv_data.data[-1].shape[1]
    r = int(m * k)
    if sample < 1:
        n = int(sample * n)
        indices = np.random.choice(mv_data.data[-1].shape[1], size=n, replace=False)
        data = []
        for vi in range(v):
            data.append(mv_data.data[vi][:, indices])
        if info:
            print(f"Using {sample * 100}%({n}) samples to init")
    else:
        data = mv_data.data
        if info:
            print(f"Using entire dataset ({n} samples) to init")
    np.random.seed(seed)
    A = [np.random.randn(mv_data.m_list[vi], r) for vi in range(v)]
    Z = [np.random.randn(r, n) for _ in range(v)]

    obj = []
    time_list = []
    start = time()

    for iter in range(max_iter):
        start_iter = time()
        # update Z
        for vi in range(v):
            Z[vi] = A[vi].T @ data[vi]

        # updata A
        for iv in range(v):
            Up, s, Vp = np.linalg.svd(data[iv] @ Z[iv].T, full_matrices=False)
            A[iv] = Up @ Vp

        # weight
        loss_recon_list = []
        for vi in range(v):
            loss_recon_v = np.linalg.norm(data[vi] - A[vi] @ Z[vi], 'fro') ** 2
            loss_recon_list.append(loss_recon_v)
        inverses = [1 / r for r in loss_recon_list]
        weight = [inv / sum(inverses) for inv in inverses]
        # loss
        loss = 0.
        for vi in range(v):
            loss += weight[vi] * loss_recon_list[vi]
        obj.append(loss)
        if info_iter:
            print(f"Iter {iter + 1}, loss: {obj[-1]:.5e}")
        time_list.append(time() - start_iter)

        # converge
        if iter > 2 and abs(obj[iter - 1] - obj[iter]) < 1e-8:
            break

    if info:
        print(f"Inited with {len(obj)} iters, time cost: {time() - start:.3f}s")
    if ret_anchor:
        return A
    else:
        return Z.T, time_list
