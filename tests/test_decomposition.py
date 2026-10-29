import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import decomposition as Dc  # noqa: E402


def test_pair_estimator_recovers_injected_components():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(40):
        mu = rng.normal(0, 0.05)
        for d in range(6):
            ud = rng.normal(0, 0.04)
            for t in range(3):
                ut = rng.normal(0, 0.05)
                for k in range(2):
                    uk = rng.normal(0, 0.06)
                    for _ in range(15):
                        rows.append((c, f"{c}-{d}", f"{c}-{d}-{t}", f"{c}-{d}-{t}-{k}", mu + ud + ut + uk + rng.normal(0, 0.1)))
    c, d, t, k, x = map(np.array, zip(*rows))
    est, se = Dc.decompose(c, d, t, k, x.astype(float))
    assert abs(est["class"] - 0.06 ** 2) < 0.0012
    assert abs(est["teacher"] - 0.05 ** 2) < 0.0012
    assert abs(est["district"] - 0.04 ** 2) < 0.0012
    assert all(v > 0 for v in se.values())


def test_no_context_variation_gives_zero():
    rng = np.random.default_rng(1)
    n = 6000
    c = rng.integers(0, 20, n)
    d = c * 100 + rng.integers(0, 8, n)
    t = d * 10 + rng.integers(0, 3, n)
    k = t * 10 + rng.integers(0, 2, n)
    x = 0.1 * c + rng.normal(0, 0.5, n)  # contrast means differ; no context variation
    est, se = Dc.decompose(c, d, t, k, x)
    for comp in Dc.COMPONENTS:
        assert abs(est[comp]) < 4 * se[comp]
