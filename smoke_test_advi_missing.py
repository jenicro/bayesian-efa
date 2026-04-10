"""
Smoke test: PyMC ADVI with missing data (MAR pattern).
N=50, P=6, K=2, 20% NaN, lower_triangular identification, 500 ADVI iterations.
"""

import sys
import traceback
import numpy as np

print("=== PyMC ADVI Missing-Data Smoke Test ===\n")

# ── 1. Create tiny dataset with 20% missing ──────────────────────────────────
rng = np.random.default_rng(42)
N, P, K = 50, 6, 2

# Simple true loading matrix: 3 items per factor
Lambda_true = np.array([
    [0.8, 0.0],
    [0.7, 0.0],
    [0.6, 0.0],
    [0.0, 0.8],
    [0.0, 0.7],
    [0.0, 0.6],
])
Omega_true = np.array([[1.0, 0.3], [0.3, 1.0]])

# Draw factors and noise
F = rng.multivariate_normal(np.zeros(K), Omega_true, size=N)
psi = rng.uniform(0.3, 0.6, size=P)
eps = rng.normal(0, psi, size=(N, P))
Y = F @ Lambda_true.T + eps

# Inject 20% NaN completely at random (MCAR → satisfies MAR)
mask = rng.random(size=(N, P)) < 0.20
Y[mask] = np.nan

n_missing = int(mask.sum())
pct_missing = 100 * n_missing / (N * P)
print(f"Dataset: N={N}, P={P}, K_true={K}")
print(f"Missing cells: {n_missing} / {N*P}  ({pct_missing:.1f}%)\n")

# ── 2. Build the PyMC model ──────────────────────────────────────────────────
print("Building PyMC model (lower_triangular, K=2)...")
try:
    sys.path.insert(0, "/c/Code/Fun-With-Bayes-EFA")
    from bayesian_efa import build_model

    model = build_model(
        Y,
        n_factors=2,
        identification="lower_triangular",
        standardize=True,
    )
    print("  build_model() succeeded.\n")
except Exception:
    print("ERROR during build_model():")
    traceback.print_exc()
    sys.exit(1)

# ── 3. Fit with ADVI (500 iterations) ───────────────────────────────────────
print("Fitting with ADVI (500 iterations) ...")
import pymc as pm

elbo_history = []

class _ELBOTracker:
    def __call__(self, approx, loss, i):
        # loss is an array of losses up to iteration i; last entry = current
        val = np.asarray(loss).flat[-1]
        elbo_val = -float(val)
        elbo_history.append(elbo_val)
        if (i - 1) % 100 == 0 or i == 1:
            print(f"  iter {i:4d}  ELBO = {elbo_val:12.4f}")

try:
    with model:
        approx = pm.fit(
            n=500,
            method="advi",
            callbacks=[_ELBOTracker()],
            progressbar=False,
            random_seed=7,
        )

    final_elbo = elbo_history[-1] if elbo_history else float("nan")
    print(f"\n  ADVI complete.  Final ELBO = {final_elbo:.4f}")
    print("\n=== RESULT: SUCCESS ===")

except Exception:
    print("\nERROR during ADVI fit:")
    traceback.print_exc()
    print("\n=== RESULT: FAILED ===")
    sys.exit(1)
