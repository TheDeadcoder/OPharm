import numpy as np


def cluster_draws(strata, n=10000, seed=0):
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(g, (n, len(g))) for g in groups], axis=1)


def cluster_ci(values, strata, n=10000, seed=0, level=0.95):
    values = np.asarray(values, dtype=float)
    stats = values[cluster_draws(strata, n, seed)].mean(1)
    lo, hi = np.percentile(stats, [50 * (1 - level), 50 * (1 + level)])
    return float(values.mean()), float(lo), float(hi)


def cluster_ratio_ci(num, den, strata, n=10000, seed=0, level=0.95):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    d = cluster_draws(strata, n, seed)
    stats = num[d].mean(1) / den[d].mean(1)
    lo, hi = np.percentile(stats, [50 * (1 - level), 50 * (1 + level)])
    return float(num.mean() / den.mean()), float(lo), float(hi)


def p_beyond(draws, null, side):
    bad = np.asarray(draws) <= null if side == "greater" else np.asarray(draws) >= null
    return float((bad.sum() + 1) / (len(bad) + 1))


def holm(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    adj, run = np.empty(len(pvals)), 0.0
    for i, k in enumerate(np.argsort(pvals)):
        run = max(run, min(1.0, (len(pvals) - i) * pvals[k]))
        adj[k] = run
    return adj, adj <= alpha
