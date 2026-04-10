"""Quick smoke test for the Stan model: small N, P=6, K=2."""
import numpy as np
from bayesian_efa import simulate_data
from befa_stan import fit_stan

if __name__ == "__main__":
    Y, L_true, Om_true = simulate_data(N=150, P=6, K=2, seed=2)
    print("Data shape:", Y.shape)
    print("Compiling + sampling…")
    fit = fit_stan(
        Y, K=2,
        chains=2, iter_warmup=300, iter_sampling=300, seed=2,
    )
    print(fit.diagnose())
    Lambda = fit.stan_variable("Lambda")  # (draws, P, K)
    Omega = fit.stan_variable("Omega")    # (draws, K, K)
    L_mean = Lambda.mean(0)
    O_mean = Omega.mean(0)

    # standardize ground truth same way the model does (Y was standardized)
    col_sd = Y.std(0, ddof=1)
    L_true_std = L_true / col_sd[:, None]

    print("\nLambda posterior mean:")
    print(np.round(L_mean, 2))
    print("Lambda truth (std):")
    print(np.round(L_true_std, 2))
    print("\nOmega posterior mean:")
    print(np.round(O_mean, 2))
    print("Omega truth:")
    print(np.round(Om_true, 2))

    summ = fit.summary()
    rows = summ.loc[summ.index.str.startswith(("Lambda", "Omega", "psi"))]
    print("\nR-hat: max =", rows["R_hat"].max(),
          "  min ESS_bulk =", rows["ESS_bulk"].min())
