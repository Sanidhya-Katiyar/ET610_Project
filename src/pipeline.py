"""One cross-fitted run of the pre-specified analysis, plus summaries.

Used identically by the outcome-blind simulation (synthetic y) and the confirmatory analysis (real y).
"""
import numpy as np
import pandas as pd

import estimation as E

MODELS = ("M0", "MB", "MBT", "MS", "MST")


def model_columns(xb, xt, score_col="s"):
    return {"M0": [], "MB": list(xb), "MBT": list(xb) + list(xt), "MS": [score_col], "MST": [score_col] + list(xt)}


def crossfit(df, xb, xt, seed, n_folds=5, weight_col=None, nuisance_lam=10.0, score_col="s", models=MODELS,
             p_mode="train", group_col="group", tau_learner=None):
    """df columns: contrast_id, z, y, group, xb..., xt..., score_col. Returns row-level held-out frame.

    p_mode: "train" = arm share of the contrast in training rows (primary); "all" = arm share in all rows of
    the contrast (design quantity, uses Z only); "half" = fixed 1/2.
    group_col: column whose values are kept together in folds (primary: student-class components).
    tau_learner: optional callable(Xtr, phi_tr, codes_tr, n_codes, groups_tr, seed) -> object with
    .predict(X, codes); replaces the pooled ridge (used for the flexible-learner sensitivity)."""
    cols = model_columns(xb, xt, score_col)
    nuis_cols = list(dict.fromkeys(list(xb) + list(xt) + [score_col]))
    codes, n_codes = E.contrast_index(df.contrast_id.to_numpy())
    y = df.y.to_numpy(float)
    z = df.z.to_numpy(int)
    groups = df[group_col].to_numpy()
    w_all = df[weight_col].to_numpy(float) if weight_col else np.ones(len(df))
    folds = E.make_folds(groups, n_folds, seed)
    out = pd.DataFrame(index=df.index)
    out["fold"] = folds
    for k in ["phi", "p", "mu0", "mu1"] + [f"tau_{m}" for m in models]:
        out[k] = np.nan
    lam_log = []
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        std = E.Standardizer().fit(df.loc[tr, nuis_cols].to_numpy(float))
        Xall = std.transform(df[nuis_cols].to_numpy(float))
        colidx = {c: i for i, c in enumerate(nuis_cols)}
        if p_mode == "train":
            p_c, _ = E.arm_share(z[tr], codes[tr], n_codes)
        elif p_mode == "all":
            p_c, _ = E.arm_share(z, codes, n_codes)
        elif p_mode == "half":
            p_c = np.full(n_codes, 0.5)
        else:
            raise ValueError(p_mode)
        p_c = np.clip(p_c, 0.02, 0.98)
        # Held-out pseudo-outcome: nuisances trained on all training rows.
        nu = E.WithinLinearOutcome(nuisance_lam).fit(Xall[tr], y[tr], z[tr], codes[tr], n_codes)
        mu0, mu1 = nu.predict(Xall[te], codes[te])
        p_te = p_c[codes[te]]
        out.loc[te, "phi"] = E.dr_score(y[te], z[te], p_te, mu0, mu1)
        out.loc[te, ["p", "mu0", "mu1"]] = np.c_[p_te, mu0, mu1]
        # Training pseudo-outcome: inner 2-fold cross-fitting of the nuisances.
        tr_idx = np.where(tr)[0]
        inner = E.make_folds(groups[tr_idx], 2, seed + 1000 + f)
        phi_tr = np.empty(len(tr_idx))
        for h in (0, 1):
            a, b = tr_idx[inner != h], tr_idx[inner == h]
            nu_h = E.WithinLinearOutcome(nuisance_lam).fit(Xall[a], y[a], z[a], codes[a], n_codes)
            m0, m1 = nu_h.predict(Xall[b], codes[b])
            phi_tr[inner == h] = E.dr_score(y[b], z[b], p_c[codes[b]], m0, m1)
        for m in models:
            idx = [colidx[c] for c in cols[m]]
            Xtr = Xall[tr][:, idx] if idx else None
            Xte = Xall[te][:, idx] if idx else None
            if tau_learner is not None and m != "M0":
                model, lam = tau_learner(Xtr, phi_tr, codes[tr], n_codes, groups[tr], seed + 77 + f), (np.nan, np.nan)
            else:
                model, lam = E.tune_and_fit(Xtr, phi_tr, codes[tr], n_codes, groups[tr], seed + 77 + f, w=w_all[tr])
            out.loc[te, f"tau_{m}"] = model.predict(Xte, codes[te])
            lam_log.append({"fold": f, "model": m, "lam1": lam[0], "lam2": lam[1]})
    return out, pd.DataFrame(lam_log)


def risk_diff(out, df, a, b, weights="student"):
    """Held-out tau-risk difference R(a) - R(b) (positive = b better) and cluster SE."""
    d = (out.phi - out[f"tau_{a}"]) ** 2 - (out.phi - out[f"tau_{b}"]) ** 2
    w = _weights(df, weights)
    return E.cluster_mean_se(d.to_numpy(), df.group.to_numpy(), w)


def _weights(df, weights):
    if weights == "student":
        return np.ones(len(df))
    if weights == "contrast":
        n = df.groupby("contrast_id").contrast_id.transform("size").to_numpy(float)
        return 1.0 / n
    raise ValueError(weights)


def policy_gain(out, df, a, b, weights="student"):
    """Held-out AIPW value of policy from model b minus policy from model a (a may be 'fixed')."""
    y, z = df.y.to_numpy(float), df.z.to_numpy(int)
    p, mu0, mu1 = out.p.to_numpy(), out.mu0.to_numpy(), out.mu1.to_numpy()
    pol = lambda m: (out[f"tau_{m}"].to_numpy() > 0).astype(int)
    pa = pol("M0") if a == "fixed" else pol(a)
    pb = pol(b)
    d = E.policy_score(y, z, p, mu0, mu1, pb) - E.policy_score(y, z, p, mu0, mu1, pa)
    return E.cluster_mean_se(d, df.group.to_numpy(), _weights(df, weights))
