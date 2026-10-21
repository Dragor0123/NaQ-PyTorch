"""Token candidates (sgc / pprrow top-50) and per-episode token sets [E,5,m] + mask (CLAUDE.md §7). No rng, no labels."""
import numpy as np

from cape.generators import N_QRY, N_WAY, top_in_pool

NCAND = 50


def load_candidates(ds, tok, nbr=None, cue=None):
    """-> [N, 50] candidate ids sorted by ranker score (desc), -1 = none."""
    if tok == "sgc":
        if nbr is None:
            z = np.load(f"cache/{ds}/pool.npz"); nbr, cue = z["nbr"], z["cue"]
        sc = np.where(nbr >= 0, cue[:, :, 1], -np.inf)
        o = np.argsort(-sc, axis=1, kind="stable")[:, :NCAND]
        top = np.take_along_axis(nbr, o, 1).astype(np.int64)
        return np.where(np.take_along_axis(sc, o, 1) > -np.inf, top, -1)
    if tok == "pprrow":
        return np.load(f"cache/{ds}/pprrow_top50.npz")["nbr"].astype(np.int64)
    raise ValueError(tok)


def token_sets(cand, spt, qry, m):
    """cand [N,50]; spt [E,5]; qry [E,50] -> tok [E,5,m] (0 where masked), mask [E,5,m].
    Excludes the episode's 50 queries and its 5 anchors; keeps the first m remaining candidates."""
    E = len(spt)
    tok, mask = np.zeros((E, N_WAY, m), np.int64), np.zeros((E, N_WAY, m), bool)
    for s in range(0, E, 500):
        c = cand[spt[s:s + 500]]                                          # [e,5,50]
        ex = np.concatenate([qry[s:s + 500], spt[s:s + 500]], 1)          # [e,55]
        ok = (c >= 0) & ~(c[:, :, :, None] == ex[:, None, None, :]).any(3)
        o = np.argsort(~ok, axis=2, kind="stable")[:, :, :m]
        tok[s:s + 500] = np.where(np.take_along_axis(ok, o, 2), np.take_along_axis(c, o, 2), 0)
        mask[s:s + 500] = np.take_along_axis(ok, o, 2)
    return tok, mask


def token_stats(tok, mask, spt, y):
    """Analysis only (labels)."""
    n = mask.sum(2)
    ya = y[spt][:, :, None]
    return {"shortfall_rate": float((n < mask.shape[2]).mean()), "empty_rate": float((n == 0).mean()),
            "mean_tokens": float(n.mean()), "token_purity": float((y[tok] == ya)[mask].mean())}
