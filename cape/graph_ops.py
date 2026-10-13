"""Normalized adjacency, truncated PPR columns, SGC / raw dense features (torch)."""
import torch

from cape.data import norm_adj, to_torch_sparse


def adj_hat_torch(A, device):
    return to_torch_sparse(norm_adj(A), device)


def ppr_columns(Ahat, idx, alpha=0.1, K=50):
    """Pi[:, idx] = alpha * sum_{k=0..K} (1-alpha)^k Ahat^k E_idx  -> [N, len(idx)]."""
    N = Ahat.shape[0]
    idx = torch.as_tensor(idx, device=Ahat.device)
    Z = torch.zeros(N, len(idx), device=Ahat.device)
    Z[idx, torch.arange(len(idx), device=Ahat.device)] = 1.0
    acc = alpha * Z
    for k in range(1, K + 1):
        Z = torch.sparse.mm(Ahat, Z)
        acc.add_(Z, alpha=alpha * (1 - alpha) ** k)
    return acc


def _l2_rows_(M):
    return M.div_(M.norm(dim=1, keepdim=True).clamp_min(1e-12))


def sgc_features(Ahat, X):
    """Ahat^2 X (dense), row L2-normalized. X: torch sparse [N, d]."""
    T = torch.sparse.mm(Ahat, X).to_dense()
    S = torch.sparse.mm(Ahat, T)
    del T
    torch.cuda.empty_cache()
    return _l2_rows_(S)


def row_normalize_dense(X):
    return _l2_rows_(X.to_dense())
