"""Design-based decomposition of treatment-effect variation into classroom, teacher and district
components, using products of DR pseudo-outcomes of *different* students in the same contrast.

Students are randomized independently, so for i != j in the same contrast (with nuisances that use
neither i's nor j's outcome), E[phi_i * phi_j] = E[delta_i * delta_j]. With classes nested in teachers
nested in districts, and pair types
    same class (cl), same teacher / different class (t'), same district / different teacher (d'),
    different district (0),
the within-contrast differences of mean pair products give
    V_class    = M_cl - M_t',   V_teacher = M_t' - M_d',   V_district = M_d' - M_0,
i.e. the variances of the class, teacher and district components of the contrast-level effect. The
contrast's average effect cancels in each difference. Contrast-level values are pooled with weights
n_c; standard errors come from a leave-one-district-out jackknife.
"""
import numpy as np
import pandas as pd

COMPONENTS = ("class", "teacher", "district")


def _pair_sums(s, q):
    return (s * s - q) / 2.0


def district_aggregates(contrast, district, teacher, klass, x):
    """Per (contrast, district) sufficient statistics: within-class / -teacher / -district pair sums and
    counts, plus the district's sum, sum of squares and size (for the cross-district term)."""
    d = pd.DataFrame({"c": contrast, "d": district, "t": teacher, "k": klass, "x": x, "x2": np.asarray(x) ** 2})
    cl = d.groupby(["c", "d", "t", "k"], sort=False).agg(s=("x", "sum"), q=("x2", "sum"), n=("x", "size"))
    cl["a"], cl["p"] = _pair_sums(cl.s, cl.q), cl.n * (cl.n - 1) / 2
    te = cl.groupby(level=["c", "d", "t"]).agg(s=("s", "sum"), q=("q", "sum"), n=("n", "sum"), a_cl=("a", "sum"), p_cl=("p", "sum"))
    te["a_t"], te["p_t"] = _pair_sums(te.s, te.q), te.n * (te.n - 1) / 2
    di = te.groupby(level=["c", "d"]).agg(s=("s", "sum"), q=("q", "sum"), n=("n", "sum"), a_cl=("a_cl", "sum"),
                                          p_cl=("p_cl", "sum"), a_t=("a_t", "sum"), p_t=("p_t", "sum"))
    di["a_d"], di["p_d"] = _pair_sums(di.s, di.q), di.n * (di.n - 1) / 2
    return di.reset_index()


def _contrast_components(g):
    """g: per-contrast totals with columns s, q, n, a_cl, p_cl, a_t, p_t, a_d, p_d. Returns V's (NaN where undefined)."""
    a_all, p_all = _pair_sums(g.s, g.q), g.n * (g.n - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        m_cl = g.a_cl / g.p_cl
        m_t = (g.a_t - g.a_cl) / (g.p_t - g.p_cl)
        m_d = (g.a_d - g.a_t) / (g.p_d - g.p_t)
        m_0 = (a_all - g.a_d) / (p_all - g.p_d)
    out = pd.DataFrame({"class": m_cl - m_t, "teacher": m_t - m_d, "district": m_d - m_0, "n": g.n}, index=g.index)
    return out.replace([np.inf, -np.inf], np.nan)


def _pool(v):
    res = {}
    for comp in COMPONENTS:
        ok = v[comp].notna()
        res[comp] = float(np.sum(v.loc[ok, comp] * v.loc[ok, "n"]) / np.sum(v.loc[ok, "n"])) if ok.any() else np.nan
    res["total_context"] = res["class"] + res["teacher"] + res["district"]
    return res


def decompose(contrast, district, teacher, klass, x, jackknife=True):
    """Pooled variance components (and leave-one-district-out jackknife SEs)."""
    di = district_aggregates(contrast, district, teacher, klass, x)
    cols = ["s", "q", "n", "a_cl", "p_cl", "a_t", "p_t", "a_d", "p_d"]
    tot = di.groupby("c")[cols].sum()
    est = _pool(_contrast_components(tot))
    if not jackknife:
        return est, None
    reps = []
    for dist, part in di.groupby("d"):
        t2 = tot.copy()
        sub = part.set_index("c")[cols]
        t2.loc[sub.index, cols] = t2.loc[sub.index, cols] - sub
        t2 = t2[t2.n > 0]
        reps.append(_pool(_contrast_components(t2)))
    reps = pd.DataFrame(reps)
    G = len(reps)
    se = {k: float(np.sqrt((G - 1) / G * np.sum((reps[k] - reps[k].mean()) ** 2))) for k in reps.columns}
    return est, se


def nest_hierarchy(klass, teacher, district):
    """Force strict nesting: each class gets its modal teacher, each teacher its modal district."""
    d = pd.DataFrame({"k": klass, "t": teacher, "d": district})
    kt = d.groupby("k").t.agg(lambda s: s.mode().iloc[0])
    t = d.k.map(kt)
    td = pd.DataFrame({"t": t, "d": d.d}).groupby("t").d.agg(lambda s: s.mode().iloc[0])
    return t.to_numpy(), t.map(td).to_numpy()
