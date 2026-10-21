"""TP loss (a2 / b) with query-aligned prototypes, and attention diagnostics (CLAUDE.md §7)."""
import torch
import torch.nn.functional as F


def tp_attention(hq, ht, mask, tau, attn="softmax"):
    """hq [Q,D], ht [C,m,D], mask [C,m] -> alpha [Q,C,m] (0 on masked; all-0 row if no tokens).
    attn="mean": uniform over valid tokens (TP02 control, tau ignored)."""
    if attn == "mean":
        w = mask.to(hq.dtype) / mask.sum(1, keepdim=True).clamp_min(1)
        return w[None].expand(hq.shape[0], -1, -1)
    s = torch.einsum("qd,cmd->qcm", F.normalize(hq, dim=-1), F.normalize(ht, dim=-1)) / tau
    s = s.masked_fill(~mask[None], -1e9)
    return torch.softmax(s, dim=-1) * mask[None]


def tp_logits(hq, ha, ht, mask, lam, tau, neg, yq, attn="softmax"):
    """ha [C,D] anchors. Prototype t^c_q = (1-lam) h_a + lam sum_j alpha h_j; anchor only if T(a_c) is empty."""
    alpha = tp_attention(hq, ht, mask, tau, attn)
    agg = torch.einsum("qcm,cmd->qcd", alpha, ht)
    has = mask.any(1).to(hq.dtype)[None, :, None]                       # [1,C,1]
    t = ha[None] + lam * has * (agg - ha[None])                         # [Q,C,D]
    if neg == "b":                                                      # attention on the positive class only
        pos = F.one_hot(yq, ha.shape[0]).to(hq.dtype)[:, :, None]
        t = pos * t + (1 - pos) * ha[None]
    elif neg != "a2":
        raise ValueError(neg)
    return -((hq[:, None, :] - t) ** 2).sum(-1)


def tp_loss(hq, ha, ht, mask, lam, tau, neg, yq, attn="softmax"):
    return F.cross_entropy(tp_logits(hq, ha, ht, mask, lam, tau, neg, yq, attn), yq)


def naq_loss(hq, ha, yq):
    return F.cross_entropy(-torch.cdist(hq, ha) ** 2, yq)


@torch.no_grad()
def attn_diag(hq, ht, mask, tau, yq, tok_lab, anc_lab, q_lab, attn="softmax"):
    """Own-class tokens of each query. tok_lab [C,m], anc_lab [C], q_lab [Q] (analysis only).
    Returns per-query lists: normalized entropy (|T|>=2), mass on tokens with the anchor's label,
    uniform expectation (fraction of such tokens), mass on tokens with the query's label."""
    alpha = tp_attention(hq, ht, mask, tau, attn)[torch.arange(len(yq)), yq]   # [Q,m]
    mk = mask[yq]                                                        # [Q,m]
    n = mk.sum(1)
    sel = n >= 1
    ent = -(alpha * torch.log(alpha.clamp_min(1e-30))).sum(1)
    same_a = (tok_lab[yq] == anc_lab[yq][:, None]) & mk
    same_q = (tok_lab[yq] == q_lab[:, None]) & mk
    return {"norm_entropy": (ent / torch.log(n.clamp_min(2).float()))[n >= 2].tolist(),
            "mass_anchor_label": (alpha * same_a).sum(1)[sel].tolist(),
            "uniform_anchor_label": (same_a.sum(1).float() / n.clamp_min(1))[sel].tolist(),
            "mass_query_label": (alpha * same_q).sum(1)[sel].tolist()}
