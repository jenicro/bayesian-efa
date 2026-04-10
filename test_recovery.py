"""Recovery test: simulate, fit (ADVI + NUTS), align via Procrustes,
   compare estimated Lambda/Omega to truth."""
import numpy as np
from bayesian_efa import simulate_data, fit_fast, fit_full


def procrustes_align(L_est, L_true):
    """Find signed permutation P minimizing ||L_est P - L_true||."""
    K = L_true.shape[1]
    # use linear assignment on |corr|
    from itertools import permutations
    best = None
    best_err = np.inf
    # try all perms (K small)
    for perm in permutations(range(K)):
        Lp = L_est[:, list(perm)]
        # sign per column
        signs = np.sign((Lp * L_true).sum(0))
        signs[signs == 0] = 1
        Lp = Lp * signs
        err = np.linalg.norm(Lp - L_true)
        if err < best_err:
            best_err = err
            best = (list(perm), signs)
    perm, signs = best
    return perm, signs, best_err


def evaluate(name, L_post, Om_post, L_true, Om_true):
    perm, signs, err = procrustes_align(L_post, L_true)
    L_aligned = L_post[:, perm] * signs
    # reorder Omega same way
    Om_aligned = Om_post[np.ix_(perm, perm)] * np.outer(signs, signs)
    frob_L = np.linalg.norm(L_aligned - L_true)
    max_L = np.abs(L_aligned - L_true).max()
    frob_O = np.linalg.norm(Om_aligned - Om_true)
    print(f"\n=== {name} ===")
    print(f"  Lambda Frobenius err: {frob_L:.3f}   max|err|: {max_L:.3f}")
    print(f"  Omega  Frobenius err: {frob_O:.3f}")
    print("  Lambda_true:")
    print(np.round(L_true, 2))
    print("  Lambda_est (aligned):")
    print(np.round(L_aligned, 2))
    print("  Omega_true:")
    print(np.round(Om_true, 2))
    print("  Omega_est (aligned):")
    print(np.round(Om_aligned, 2))
    return frob_L, frob_O


if __name__ == "__main__":
    Y, L_true, Om_true = simulate_data(N=300, P=9, K=3, seed=1)
    # standardized truth (since model standardizes Y)
    col_sd = Y.std(0, ddof=1)
    L_true_std = L_true / col_sd[:, None]

    print("Fitting ADVI…")
    idata_f, _, approx = fit_fast(Y, n_factors=3, n_iter=40_000, seed=1)
    Lf = idata_f.posterior["Lambda"].mean(("chain", "draw")).values
    Of = idata_f.posterior["Omega"].mean(("chain", "draw")).values
    evaluate("ADVI", Lf, Of, L_true_std, Om_true)

    print("\nFitting NUTS (this is slow)…")
    idata_n, _ = fit_full(Y, n_factors=3, draws=500, tune=500, chains=2, seed=1)
    Ln = idata_n.posterior["Lambda"].mean(("chain", "draw")).values
    On = idata_n.posterior["Omega"].mean(("chain", "draw")).values
    evaluate("NUTS", Ln, On, L_true_std, Om_true)

    import arviz as az
    print("\nNUTS R-hat summary:")
    s = az.summary(idata_n, var_names=["Lambda", "Omega", "psi"])
    print(s[["mean", "sd", "r_hat", "ess_bulk"]].describe())
