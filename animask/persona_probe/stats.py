"""Pure-Python statistics for the persona probe (no numpy/scipy).

Implements: Theil–Sen slope, Mann–Kendall trend test (tie-corrected normal
approximation), trajectory AUC (trapezoid over normalized time, SPASM-style),
clustered bootstrap percentile CI, paired permutation test, Holm correction,
Cohen's kappa, Cohen's d.
"""
import math
import random


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    m = n // 2
    return xs[m] if n % 2 else (xs[m - 1] + xs[m]) / 2


def norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def theil_sen(xs, ys):
    """Median of pairwise slopes; None with <2 distinct x."""
    slopes = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] != xs[i]:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    return median(slopes)


def mann_kendall(ys):
    """MK trend test on a sequence. Returns {s, z, p, n}. Two-sided p via
    normal approximation with tie correction; p=1.0 when n<3 or var=0."""
    n = len(ys)
    if n < 3:
        return {"s": 0, "z": 0.0, "p": 1.0, "n": n}
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = ys[j] - ys[i]
            s += (d > 0) - (d < 0)
    # tie correction
    counts = {}
    for y in ys:
        counts[y] = counts.get(y, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var <= 0:
        return {"s": s, "z": 0.0, "p": 1.0, "n": n}
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = 2 * (1 - norm_cdf(abs(z)))
    return {"s": s, "z": round(z, 4), "p": round(min(1.0, p), 6), "n": n}


def trajectory_auc(xs, ys):
    """Trapezoidal area under y over x normalized to [0,1] (SPASM-style
    trajectory summary). Equals the time-weighted mean level of y."""
    if not xs:
        return None
    if len(xs) == 1:
        return ys[0]
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        return mean(ys)
    pts = sorted(zip(xs, ys))
    area = 0.0
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        area += (xb - xa) / (x1 - x0) * (ya + yb) / 2
    return area


def bootstrap_ci_clustered(clusters, stat=mean, n_boot=1000, seed=0,
                           alpha=0.05):
    """clusters: list of lists of values; resample clusters with replacement,
    pool values, compute stat. Returns {stat, lo, hi, n_boot} (percentile CI).
    """
    clusters = [c for c in clusters if c]
    if not clusters:
        return {"stat": None, "lo": None, "hi": None, "n_boot": 0}
    rng = random.Random(seed)
    point = stat([v for c in clusters for v in c])
    draws = []
    k = len(clusters)
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(k)] for _ in range(k)]
        draws.append(stat([v for c in sample for v in c]))
    draws.sort()
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"stat": point, "lo": lo, "hi": hi, "n_boot": n_boot}


def paired_permutation(xs, ys, n_perm=2000, seed=0):
    """Two-sided paired permutation test on mean(x - y) via sign flips."""
    diffs = [x - y for x, y in zip(xs, ys)]
    if not diffs:
        return {"mean_diff": None, "p": None, "n": 0}
    obs = mean(diffs)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        perm = mean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(perm) >= abs(obs) - 1e-12:
            hits += 1
    return {"mean_diff": obs, "p": (hits + 1) / (n_perm + 1), "n": len(diffs)}


def holm(pvals):
    """Holm step-down adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cohens_kappa(a, b):
    """Cohen's kappa for two aligned categorical rating lists."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for c in cats:
        pe += (sum(1 for x in a if x == c) / n) * \
              (sum(1 for y in b if y == c) / n)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def percent_agreement(a, b):
    if not a or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_d(xs, ys):
    """Cohen's d with pooled SD; None if either group has <2 values."""
    xs, ys = list(xs), list(ys)
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs) / (len(xs) - 1)
    vy = sum((y - my) ** 2 for y in ys) / (len(ys) - 1)
    pooled = ((len(xs) - 1) * vx + (len(ys) - 1) * vy) / \
             (len(xs) + len(ys) - 2)
    if pooled <= 0:
        return 0.0
    return (mx - my) / math.sqrt(pooled)


def series_stats(xs, ys):
    """Bundle used by every layer for a score trajectory."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return {"n": 0, "mean": None, "theil_sen_slope": None,
                "mk": {"s": 0, "z": 0.0, "p": 1.0, "n": 0}, "auc": None,
                "total_change": None}
    pxs = [p[0] for p in pairs]
    pys = [p[1] for p in pairs]
    slope = theil_sen(pxs, pys)
    span = (max(pxs) - min(pxs)) if len(pxs) > 1 else 0
    return {"n": len(pys), "mean": mean(pys),
            "theil_sen_slope": slope,
            "mk": mann_kendall(pys),
            "auc": trajectory_auc(pxs, pys),
            "total_change": (slope * span) if slope is not None else None}
