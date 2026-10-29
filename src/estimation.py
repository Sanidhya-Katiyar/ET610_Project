"""Core estimation utilities: folds, DR pseudo-outcomes, partially pooled ridge CATE models,
held-out tau-risk differences, policy values, and cluster-robust inference.

Identification note (see protocol/ANALYSIS_PROTOCOL.md §6): within contrast c, assignment Z is
Bernoulli with a constant probability p_c (independent, one-at-a-time randomization on E-TRIALS).
For any nuisance functions mu0, mu1 fitted on data not containing unit i, the score

    phi_i = mu1(X_i) - mu0(X_i) + Z_i (Y_i - mu1(X_i)) / p_c - (1 - Z_i)(Y_i - mu0(X_i)) / (1 - p_c)

satisfies E[phi_i | X_i, training data] = tau_c(X_i). Hence for any CATE estimate tau_hat trained
without unit i, E[(phi_i - tau_hat(X_i))^2] = E[(tau_c(X_i) - tau_hat(X_i))^2] + E[Var(phi_i | X_i)], and
the second term does not depend on tau_hat. Paired held-out loss differences between two CATE
estimators evaluated against the SAME phi are therefore unbiased for their tau-risk difference.
"""
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------- folds
def make_folds(groups, n_folds, seed):
    """Assign whole groups to folds, greedily balancing row counts. Returns fold id per row."""
    rng = np.random.default_rng(seed)
    g = pd.Series(groups)
    sizes = g.value_counts()
    order = sizes.index.to_numpy().copy()
    rng.shuffle(order)
    order = sorted(order, key=lambda k: -sizes[k])  # stable sort keeps random tie order
    load = np.zeros(n_folds)
    assign = {}
    for k in order:
        f = int(np.argmin(load + rng.random(n_folds) * 1e-9))
        assign[k] = f
        load[f] += sizes[k]
    return g.map(assign).to_numpy()


# --------------------------------------------------------------------------------- preprocessing
class Standardizer:
    """Median imputation + standardisation using training rows only."""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), self.med, X)
        self.mu = Xi.mean(axis=0)
        self.sd = Xi.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        Xi = np.where(np.isnan(X), self.med, X)
        return (Xi - self.mu) / self.sd


def contrast_index(cid):
    codes, uniq = pd.factorize(cid)
    return codes, len(uniq)


def group_means(v, codes, n_codes, w=None):
    w = np.ones_like(v, dtype=float) if w is None else w
    num = np.bincount(codes, weights=v * w, minlength=n_codes)
    den = np.bincount(codes, weights=w, minlength=n_codes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, den


# ------------------------------------------------------------------------------------ nuisances
class WithinLinearOutcome:
    """mu_z(x) = ybar_{c,z} + (x - xbar_{c,z}) beta_z: arm-specific linear probability model with
    contrast-by-arm fixed effects and ridge-penalised pooled slopes (closed form)."""

    def __init__(self, lam=10.0):
        self.lam = lam

    def fit(self, X, y, z, codes, n_codes):
        self.params = {}
        for arm in (0, 1):
            m = z == arm
            ym, cnt = group_means(y[m], codes[m], n_codes)
            sums = np.zeros((n_codes, X.shape[1]))
            np.add.at(sums, codes[m], X[m])
            with np.errstate(invalid="ignore", divide="ignore"):
                Xm = sums / cnt[:, None]
            Xc = X[m] - Xm[codes[m]]
            yc = y[m] - ym[codes[m]]
            beta = np.linalg.solve(Xc.T @ Xc + self.lam * np.eye(X.shape[1]), Xc.T @ yc)
            # Contrasts with no training rows in this arm fall back to the overall arm mean.
            glob = y[m].mean()
            ym = np.where(cnt > 0, ym, glob)
            Xm = np.where(cnt[:, None] > 0, Xm, X[m].mean(axis=0))
            self.params[arm] = (ym, Xm, beta)
        return self

    def predict(self, X, codes):
        out = []
        for arm in (0, 1):
            ym, Xm, beta = self.params[arm]
            out.append(np.clip(ym[codes] + (X - Xm[codes]) @ beta, 0.0, 1.0))
        return out[0], out[1]


def dr_score(y, z, p, mu0, mu1):
    return mu1 - mu0 + z * (y - mu1) / p - (1 - z) * (y - mu0) / (1 - p)


def arm_share(z, codes, n_codes):
    share, cnt = group_means(z.astype(float), codes, n_codes)
    return np.where(cnt > 0, share, 0.5), cnt


# ------------------------------------------------------------------------------------ CATE model
class PooledRidgeCATE:
    """tau_c(x) = alpha_c + x'(beta + b_c).

    alpha_c: training mean of the pseudo-outcome in contrast c (unpenalised; = contrast ATE estimate).
    beta:    shared slopes, ridge penalty lam1 on within-contrast centred data.
    b_c:     contrast-specific deviations, ridge penalty lam2 on the residuals after the shared slopes
             (shrinks each contrast towards the shared slope).
    lam = inf switches a component off. With no features (or lam1 = lam2 = inf) this is the
    constant (contrast-ATE-only) model.

    Sufficient statistics (Gram matrices) are computed once per training set, so several penalty
    pairs can be solved cheaply (`solve`)."""

    def __init__(self, lam1=np.inf, lam2=np.inf):
        self.lam1, self.lam2 = lam1, lam2

    def stats(self, X, phi, codes, n_codes, w=None):
        w = np.ones_like(phi) if w is None else w
        self.n_codes = n_codes
        self.alpha, cnt = group_means(phi, codes, n_codes, w)
        self.alpha = np.where(cnt > 0, self.alpha, np.average(phi, weights=w))
        self.p = X.shape[1] if X is not None else 0
        if self.p == 0:
            return self
        sums = np.zeros((n_codes, self.p))
        np.add.at(sums, codes, X * w[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Xm = np.nan_to_num(sums / cnt[:, None])
        Xc = X - self.Xm[codes]
        r = phi - self.alpha[codes]
        Xw = Xc * w[:, None]
        # Per-contrast Gram matrices A_c = sum w x x' and g_c = sum w x r (centred within contrast).
        self.Ac = np.zeros((n_codes, self.p, self.p))
        self.gc = np.zeros((n_codes, self.p))
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(n_codes + 1))
        for c in range(n_codes):
            idx = order[bounds[c]:bounds[c + 1]]
            if len(idx):
                self.Ac[c] = Xw[idx].T @ Xc[idx]
                self.gc[c] = Xw[idx].T @ r[idx]
        self.A = self.Ac.sum(axis=0)
        self.g = self.gc.sum(axis=0)
        return self

    def solve(self, lam1, lam2):
        self.lam1, self.lam2 = lam1, lam2
        p = self.p
        self.beta = np.zeros(p)
        self.b = np.zeros((self.n_codes, p))
        if p == 0:
            return self
        if np.isfinite(lam1):
            self.beta = np.linalg.solve(self.A + lam1 * np.eye(p), self.g)
        if np.isfinite(lam2):
            rhs = self.gc - self.Ac @ self.beta
            self.b = np.linalg.solve(self.Ac + lam2 * np.eye(p)[None], rhs[:, :, None])[:, :, 0]
        return self

    def fit(self, X, phi, codes, n_codes, w=None):
        return self.stats(X, phi, codes, n_codes, w).solve(self.lam1, self.lam2)

    def predict(self, X, codes):
        if X is None or self.p == 0:
            return self.alpha[codes]
        Xc = X - self.Xm[codes]
        return self.alpha[codes] + (Xc * (self.beta[None, :] + self.b[codes])).sum(axis=1)


LAM1_GRID = [np.inf, 3e4, 3e3, 3e2, 3e1]
LAM2_GRID = [np.inf, 3e3, 3e2, 3e1]


def tune_and_fit(X, phi, codes, n_codes, groups, seed, inner_folds=3, w=None):
    """Choose (lam1, lam2) by inner group-fold CV on the pseudo-outcome loss, then refit."""
    if X is None or X.shape[1] == 0:
        return PooledRidgeCATE().fit(None, phi, codes, n_codes, w), (np.inf, np.inf)
    w = np.ones_like(phi) if w is None else w
    folds = make_folds(groups, inner_folds, seed)
    losses = {(l1, l2): 0.0 for l1 in LAM1_GRID for l2 in LAM2_GRID}
    for f in range(inner_folds):
        tr, te = folds != f, folds == f
        m = PooledRidgeCATE().stats(X[tr], phi[tr], codes[tr], n_codes, w[tr])
        for key in losses:
            m.solve(*key)
            losses[key] += np.sum(w[te] * (phi[te] - m.predict(X[te], codes[te])) ** 2)
    best = min(losses, key=losses.get)  # first minimum in grid order (strongest penalties listed first)
    return PooledRidgeCATE(*best).fit(X, phi, codes, n_codes, w), best


# ------------------------------------------------------------------------------------- inference
def cluster_mean_se(d, groups, w=None):
    """Weighted mean of d and its cluster-robust (CR1-type) standard error."""
    w = np.ones_like(d) if w is None else w
    W = w.sum()
    m = np.sum(w * d) / W
    gcodes, G = contrast_index(groups)
    s = np.bincount(gcodes, weights=w * (d - m), minlength=G)
    se = np.sqrt(G / (G - 1) * np.sum(s ** 2)) / W
    return m, se


def policy_score(y, z, p, mu0, mu1, pi):
    """AIPW score for the value of policy pi (1 = assign arm 1)."""
    mu_pi = np.where(pi == 1, mu1, mu0)
    prob = np.where(pi == 1, p, 1 - p)
    return mu_pi + (z == pi) * (y - np.where(z == 1, mu1, mu0)) / prob


def blp_increment(phi, t_base, t_ext, codes, n_codes, groups, w=None):
    """Within-contrast regression phi ~ t_base + (t_ext - t_base); returns coef and cluster SE of the
    increment term (a calibration-type diagnostic, cf. Chernozhukov et al.)."""
    w = np.ones_like(phi) if w is None else w
    cols = [t_base, t_ext - t_base]
    Xd = np.vstack([c - group_means(c, codes, n_codes, w)[0][codes] for c in cols]).T
    yd = phi - group_means(phi, codes, n_codes, w)[0][codes]
    A = (Xd * w[:, None]).T @ Xd
    if np.linalg.matrix_rank(A) < 2:  # e.g. both models tuned to contrast ATEs only: not estimable
        return np.full(2, np.nan), np.full(2, np.nan)
    beta = np.linalg.solve(A, (Xd * w[:, None]).T @ yd)
    e = yd - Xd @ beta
    gcodes, G = contrast_index(groups)
    S = np.vstack([np.bincount(gcodes, weights=w * e * Xd[:, j], minlength=G) for j in range(2)]).T
    Ainv = np.linalg.inv(A)
    V = G / (G - 1) * Ainv @ (S.T @ S) @ Ainv
    return beta, np.sqrt(np.diag(V))
