import torch
import torch.nn.functional as F


def con_inter_view(features, metric='cosine', device='cpu', gamma=1, temperature=1, mask=None):
    """
    Compute inter-view contrastive loss for multi-view data.
    Parameters:
    -----------
    features: List[torch.Tensor]
        List of features from different views. Shape: (num_samples, feature_dim)

    Returns: torch.Tensor Inter-view contrastive loss value
    """
    num_view = len(features)
    num_sample = features[0].shape[0]  # n: number of samples
    features = torch.cat(features, dim=0)  # shape: (num_smp * num_view, feature_dim)
    features = F.normalize(features, dim=1)  # each row becomes unit norm

    # Construct positive-pair mask
    mask = torch.eye(num_sample, dtype=torch.float32).to(device)
    mask = mask.repeat(num_view, num_view)
    logits_mask = torch.ones_like(mask).to(device)
    logits_mask.fill_diagonal_(0)
    mask = mask * logits_mask  # Only positive pairs (same index across views) remain 1

    # Compute inter-view contrastive loss
    if metric == 'cosine':
        sim_matrix = torch.matmul(features, features.T).to(device)  # Cosine similarity in [-1, 1]
    elif metric == 'euclidean':
        dist_matrix = torch.cdist(features, features, p=2).to(device)  # Euclidean distance
        sim_matrix = torch.exp(-gamma * dist_matrix ** 2).to(device)  # Apply the exponential function
    else:
        raise ValueError("Invalid distance metric. Choose 'cosine' or 'euclidean'.")
    # Calculate pairwise similarity (logits) by dividing the similarity matrix by temperature for scaling.
    logits = sim_matrix / temperature
    # Compute log probabilities using the softmax-like formulation with masking to exclude diagonal (self-similarity).
    log_prob = logits - torch.log((torch.exp(logits) * logits_mask).sum(1, keepdim=True) + 1e-12)
    # Calculate the mean log probability for positive pairs (same sample across different views).
    mean_log_prob_pos = - (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    # Compute the final contrastive loss as the average of the positive pair losses across all samples.
    loss_con = mean_log_prob_pos.mean()

    return loss_con


def con_inter_view_n(features, metric='cosine', mask=None, temperature=1.0, gamma=1.0, device='cpu'):
    if device is None:
        device = features[0].device
    v = len(features)
    n = features[0].shape[0]

    feats = torch.cat(features, dim=0)  # (v*n, d)
    feats = F.normalize(feats, dim=1)

    if mask is None:
        base = torch.eye(n, device=device)
        mask = base.repeat(v, v)  # (v*n, v*n)
    else:
        mask = mask.to(device).float()
        assert mask.shape == (v * n, v * n), \
            f"mask shape should be {(v * n, v * n)}, got {mask.shape} (v({v}), n({n}))"

    logits_mask = torch.ones_like(mask)
    logits_mask.fill_diagonal_(0.)

    if metric == 'cosine':
        sim_matrix = feats @ feats.T  # [-1,1]
    elif metric == 'euclidean':
        dist = torch.cdist(feats, feats, p=2.)
        sim_matrix = torch.exp(-gamma * dist ** 2)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'.")

    logits = sim_matrix / temperature
    log_prob = logits - torch.log(
        (torch.exp(logits) * logits_mask).sum(1, keepdim=True) + 1e-12
    )

    mean_log_pos = - (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    return mean_log_pos.mean()


def con_loss_intra_view(features, labels, metric='cosine', device='cpu', gamma=1, temperature=1):
    """
    Compute intra-view contrastive loss for multi-view data.

    Parameters:
    -----------
    features : List[torch.Tensor]
        List of features from different views. Shape: (num_samples, feature_dim)
    labels : List[np.ndarray]
        List of pseudo-labels for each view. Each element has shape: (num_samples,)
    Returns: torch.Tensor
    Average intra-view contrastive loss across all views
    """
    num_view = len(features)
    loss_val = 0.0

    for v in range(num_view):
        feat_v, label_v = features[v], labels[v]
        feat_v = F.normalize(feat_v, dim=1)
        label_mask = (label_v.unsqueeze(0) == label_v.unsqueeze(1)).float().to(device)
        logits_mask = torch.ones_like(label_mask).fill_diagonal_(0)
        mask = label_mask * logits_mask

        if metric == 'cosine':
            sim_matrix = torch.matmul(feat_v, feat_v.T)  # Cosine similarity in [-1, 1]
        elif metric == 'euclidean':
            dist_matrix = torch.cdist(feat_v, feat_v, p=2)  # Euclidean distance
            sim_matrix = torch.exp(-gamma * dist_matrix ** 2)  # Apply exponential function
        else:
            raise ValueError("Invalid distance metric. Choose 'cosine' or 'euclidean'.")
        logits = sim_matrix / temperature
        log_prob = logits - torch.log((torch.exp(logits) * logits_mask).sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = - (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss_v = mean_log_prob_pos.mean()
        loss_val += loss_v

    loss_val /= num_view
    return loss_val


def con_reg_loss(assign_mat):
    """
    Compute entropy-based regularization loss to prevent cluster collapse.

    Args:
        assign_mat (List[Tensor]): List of (k * n) assignment matrices for each view (soft assignments over clusters).

    Returns:
        Tensor: Regularization loss scalar.
    """
    loss_val = 0.0
    for assign_mat_v in assign_mat:
        p_i = assign_mat_v.sum(1) / assign_mat_v.shape[1]  # Sum over samples -> cluster frequency
        loss_val += (p_i * torch.log(p_i + 1e-12)).sum() / len(assign_mat)  # Entropy-like penalty
    return loss_val
