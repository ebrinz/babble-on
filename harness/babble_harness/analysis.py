"""Statistics comparing conditions. Pure numpy so it runs anywhere.

Per-sample scalars come from the sampler trace; activations from the
adapter's capture. Between-condition tests: Mann-Whitney U (two-sided, normal
approximation) on scalars, and a cross-validated linear probe on pooled
activations (and on the scalar vector) with a label-permutation null.

The default probe is a mass-mean ("difference of means") classifier on
features scaled by their pooled *within-class* deviation. Scaling by the
overall deviation, as a generic standardiser does, shrinks exactly the
direction that carries the class shift; and an unregularised linear fit is
hopeless at the sample sizes this experiment will have (tens per condition
in thousands of dimensions). A ridge classifier is kept as an option.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- scalar features -------------------------------------------------------

def sample_scalars(result, denoiser=None) -> dict[str, float]:
    steps = result.steps
    n = len(steps)
    acc = np.array([s.n_accepted for s in steps], dtype=np.float64)
    ent = np.array([s.mean_entropy for s in steps], dtype=np.float64)
    stc = result.steps_to_commit()
    feats = {
        "n_steps": float(n),
        "stopped_early": float(result.stopped_early),
        "tape_consumed": float(result.tape_consumed),
        "accepted_step0": float(acc[0]) if n else 0.0,
        "accepted_mean": float(acc.mean()) if n else 0.0,
        "entropy_step0": float(ent[0]) if n else 0.0,
        "entropy_final": float(ent[-1]) if n else 0.0,
        "entropy_auc": float(ent.sum()) if n else 0.0,
        "commit_step_mean": float(stc[stc >= 0].mean()) if (stc >= 0).any() else float("nan"),
        "commit_step_spread": float(stc[stc >= 0].std()) if (stc >= 0).any() else float("nan"),
        "never_stable_frac": float((stc < 0).mean()),
        "flips": float(sum(int((s.canvas_in != s.canvas_out).sum()) for s in steps)),
    }
    caps = [s.capture for s in steps if s.capture]
    if caps and "router_entropy" in caps[0]:
        re = np.stack([c["router_entropy"] for c in caps])
        feats["router_entropy_mean"] = float(re.mean())
    if caps and "logit_lens_agree" in caps[0]:
        ll = np.stack([c["logit_lens_agree"] for c in caps])
        feats["logit_lens_depth"] = float(np.argmax(ll.mean(0) >= 0.5)) if (ll.mean(0) >= 0.5).any() else float(ll.shape[1])
        feats["logit_lens_mean"] = float(ll.mean())
    return feats


def text_scalars(ids: np.ndarray) -> dict[str, float]:
    ids = np.asarray(ids)
    n = len(ids)
    if n == 0:
        return {"distinct1": 0.0, "distinct2": 0.0, "repeat_frac": 0.0}
    d1 = len(set(ids.tolist())) / n
    bigrams = list(zip(ids[:-1].tolist(), ids[1:].tolist()))
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    rep = float((ids[1:] == ids[:-1]).mean()) if n > 1 else 0.0
    return {"distinct1": d1, "distinct2": d2, "repeat_frac": rep}


# ---- activation pooling -----------------------------------------------------

def pooled_activation(result, layer_index: int = -1, step_index: int = 0) -> np.ndarray | None:
    """Mean over canvas positions of one captured layer at one step
    (``step_index`` negative counts from the end). Returns ``[hidden]``."""
    caps = [s.capture for s in result.steps if s.capture and "hidden" in s.capture]
    if not caps:
        return None
    h = caps[step_index]["hidden"]  # [layers, L, hidden]
    return h[layer_index].mean(axis=0).astype(np.float64)


# ---- tests ---------------------------------------------------------------------

@dataclass
class MWResult:
    u: float
    z: float
    p: float
    n1: int
    n2: int
    median1: float
    median2: float


def mann_whitney(a, b) -> MWResult:
    a = np.asarray([x for x in a if not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
    b = np.asarray([x for x in b if not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return MWResult(float("nan"), float("nan"), float("nan"), n1, n2, float("nan"), float("nan"))
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    # tie-corrected variance
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    N = n1 + n2
    var = n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1))) if N > 1 else 0.0
    z = (u1 - mu) / math.sqrt(var) if var > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return MWResult(u1, z, p, n1, n2, float(np.median(a)), float(np.median(b)))


def _ridge_fit(X, y, lam=1.0):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def _ridge_predict(w, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ w


def _pooled_scale(Xtr, ytr):
    """Centre on the training mean; scale by the pooled within-class deviation."""
    mu = Xtr.mean(0)
    pos, neg = Xtr[ytr > 0], Xtr[ytr <= 0]
    var = 0.5 * (pos.var(0) + neg.var(0)) if len(pos) > 1 and len(neg) > 1 else Xtr.var(0)
    return mu, np.sqrt(var) + 1e-8


def cv_probe_accuracy(X: np.ndarray, y: np.ndarray, folds: int = 5, lam: float = 1.0, seed: int = 0,
                      method: str = "meandiff") -> float:
    """k-fold accuracy of a linear probe (labels ±1). ``method``: ``"meandiff"``
    (mass-mean, default) or ``"ridge"``. Scaling is fitted on the training
    fold only."""
    X = np.asarray(X, dtype=np.float64)
    y = np.where(np.asarray(y) > 0, 1.0, -1.0)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = max(2, min(folds, n))
    correct = 0
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        mu, sd = _pooled_scale(X[train], y[train])
        A, B = (X[train] - mu) / sd, (X[test] - mu) / sd
        if method == "ridge":
            w = _ridge_fit(A, y[train], lam)
            score = _ridge_predict(w, B)
        else:
            m1, m0 = A[y[train] > 0].mean(0), A[y[train] <= 0].mean(0)
            score = (B - (m1 + m0) / 2) @ (m1 - m0)
        pred = np.where(score >= 0, 1.0, -1.0)
        correct += int((pred == y[test]).sum())
    return correct / n


@dataclass
class ProbeResult:
    accuracy: float
    null_mean: float
    null_sd: float
    p_value: float
    n: int
    dim: int
    mean_diff_norm: float


def probe_with_permutation_null(X, y, n_perm: int = 200, seed: int = 0, **kw) -> ProbeResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    acc = cv_probe_accuracy(X, y, seed=seed, **kw)
    rng = np.random.default_rng(seed + 1)
    null = np.array([cv_probe_accuracy(X, rng.permutation(y), seed=seed, **kw) for _ in range(n_perm)])
    p = float((np.sum(null >= acc) + 1) / (n_perm + 1))
    md = X[y > 0].mean(0) - X[y <= 0].mean(0) if (y > 0).any() and (y <= 0).any() else np.zeros(X.shape[1])
    return ProbeResult(acc, float(null.mean()), float(null.std()), p, len(y), X.shape[1], float(np.linalg.norm(md)))
