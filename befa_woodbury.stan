// Bayesian EFA -- Woodbury-optimized marginalized FIML for missing data.
//
// WHY THIS MODEL EXISTS
// ---------------------
// befa_missing.stan (pattern-mixture FIML) groups rows by missingness pattern
// and performs one Cholesky decomposition of a d x d matrix per unique pattern
// per HMC step: O(n_patterns * d^3).  With P=152 and 80% missingness,
// n_patterns approaches N and d ~ 30, making each step extremely slow.
//
// This model computes the *same marginal log-likelihood* using the Woodbury
// matrix identity.  For each observed row n with observed column set S_n:
//
//   y_{n,S_n} ~ MVN(0, Sigma_{S_n})
//   Sigma_S = Lambda_S * Omega * Lambda_S' + D_S,  D_S = diag(psi_S^2)
//
// KEY IDENTITIES (all K x K -- never d x d):
//   log|Sigma_S| = log|D_S| + log|Omega| + log|M_S|       (det lemma)
//   Sigma_S^{-1} = D_S^{-1} - D_S^{-1} Lambda_S M_S^{-1} Lambda_S' D_S^{-1}
//   y' Sigma_S^{-1} y = ||y./psi_S||^2 - ||v||^2
//     where  u = Lambda_S' * (y ./ psi_S^2),  v = L_M^{-1} u
//     and    M_S = Omega^{-1} + Lambda_S' * D_S^{-1} * Lambda_S   (K x K only!)
//
// Cost per HMC step: O(n_obs * K^2 + N * K^3) -- no d x d Cholesky!
// For K=5, P=152, N=500, 80% miss: roughly 50-100x faster than pattern FIML.
//
// MATHEMATICAL EQUIVALENCE
// ------------------------
// This is NOT an approximation.  It computes the identical marginal
// log p(Y_obs | Lambda, Omega, psi) that befa_missing.stan does -- factor
// scores are analytically marginalised in both models.  The only difference
// is computational efficiency.
//
// DATA FORMAT
// -----------
// Observations stored in CSR row format, sorted by row:
//   row_len[n]   = number of observed columns in row n
//   row_start[n] = 1-indexed start in obs_col / y_vals for row n
//   obs_col[i]   = 1-indexed column of the i-th observation
//   y_vals[i]    = observed value of the i-th observation
// Rows with row_len[n] == 0 are silently skipped (no contribution to log-lik).

data {
  int<lower=1> N;          // number of observations (rows)
  int<lower=2> P;          // number of variables (columns)
  int<lower=1> K;          // number of factors

  // CSR sparse format (only observed cells, sorted row-major)
  int<lower=0>                         n_obs;
  array[N] int<lower=0>               row_len;    // # observed cols per row
  array[N] int<lower=1>               row_start;  // 1-indexed start in obs arrays
  array[n_obs] int<lower=1, upper=P>  obs_col;    // column index (1-indexed)
  vector[n_obs]                        y_vals;     // observed values

  // Loading structure (same convention as befa.stan)
  int<lower=1>   M;
  array[M]   int<lower=1>  row_idx;
  array[M]   int<lower=1>  col_idx;
  int<lower=0>   n_diag;
  array[n_diag] int<lower=1>  diag_row_idx;
  array[n_diag] int<lower=1>  diag_col_idx;

  // Hyperparameters
  real<lower=0> tau0;
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
  real<lower=0> lkj_eta;
}

parameters {
  // ---- Loading matrix -------------------------------------------------
  vector<lower=0>[n_diag] diag_loadings;
  vector[M]               z;
  vector<lower=0>[M]      lam;
  real<lower=0>           tau;
  real<lower=0>           caux;

  // ---- Uniquenesses ---------------------------------------------------
  vector<lower=0>[P]      psi;

  // ---- Factor correlation Cholesky ------------------------------------
  cholesky_factor_corr[K] L_Omega;
}

transformed parameters {
  // ---- Assemble loading matrix ----------------------------------------
  matrix[P, K] Lambda = rep_matrix(0, P, K);
  {
    real      c         = slab_scale * sqrt(caux);
    vector[M] lam_tilde =
      sqrt( (c^2 * square(lam)) ./ (c^2 + tau^2 * square(lam)) );
    vector[M] free_vals = z .* (tau * lam_tilde);
    for (k in 1:n_diag)
      Lambda[diag_row_idx[k], diag_col_idx[k]] = diag_loadings[k];
    for (m in 1:M)
      Lambda[row_idx[m], col_idx[m]] = free_vals[m];
  }
}

model {
  // ---- Regularised horseshoe (Piironen & Vehtari 2017) ----------------
  z    ~ std_normal();
  lam  ~ student_t(3, 0, 1);
  tau  ~ student_t(3, 0, tau0);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);

  // ---- Other priors ---------------------------------------------------
  diag_loadings ~ normal(0, 1);
  psi           ~ normal(0, 1);
  L_Omega       ~ lkj_corr_cholesky(lkj_eta);

  // ---- Precompute Omega^{-1} and log|Omega| from the Cholesky factor --
  // L_Omega satisfies: L_Omega * L_Omega' = Omega.
  // L_inv = L_Omega^{-1}  (lower triangular, computed by triangular solve)
  // Omega_inv = L_inv' * L_inv  (= crossprod(L_inv))
  // log|Omega| = 2 * sum(log(diag(L_Omega)))
  matrix[K, K] L_inv       = mdivide_left_tri_low(L_Omega, identity_matrix(K));
  matrix[K, K] Omega_inv   = crossprod(L_inv);
  real         log_det_Omega = 2.0 * sum(log(diagonal(L_Omega)));

  // ---- Woodbury marginalized likelihood -- one block per observed row --
  for (n in 1:N) {
    int d = row_len[n];
    if (d == 0) continue;   // fully missing row -- contributes nothing

    int s = row_start[n];
    int e = s + d - 1;

    // Extract observed variables for this row
    array[d] int  cidx  = obs_col[s:e];
    vector[d]     y_n   = y_vals[s:e];
    matrix[d, K]  Lam_n = Lambda[cidx, :];
    vector[d]     psi_n = psi[cidx];

    // M_n = Omega^{-1} + Lambda_n' * D_n^{-1} * Lambda_n   (K x K)
    // D_n = diag(psi_n^2),  D_n^{-1} = diag(1/psi_n^2)
    // Symmetrize before Cholesky: floating-point can introduce tiny asymmetry
    // in the L'DL product at extreme parameter values during early warmup,
    // causing cholesky_decompose to throw "not symmetric".  The 0.5*(M+M')
    // trick costs nothing and keeps proposals from crashing the sampler.
    vector[d]    psi2_inv = inv(square(psi_n));
    matrix[K, K] M_raw   = Omega_inv
                            + Lam_n' * diag_pre_multiply(psi2_inv, Lam_n);
    matrix[K, K] M_n     = 0.5 * (M_raw + M_raw');
    matrix[K, K] L_M     = cholesky_decompose(M_n);

    // log|Sigma_n| = log|D_n| + log|Omega| + log|M_n|
    //             = 2*sum(log(psi_n)) + log_det_Omega + 2*sum(log(diag(L_M)))
    real log_det_n = 2.0 * sum(log(psi_n))
                     + log_det_Omega
                     + 2.0 * sum(log(diagonal(L_M)));

    // Woodbury quadratic form:
    //   u = Lambda_n' * (y_n / psi_n^2)     [K-vector]
    //   v = L_M^{-1} * u                    [lower-tri solve]
    //   y' Sigma_n^{-1} y = ||y_n/psi_n||^2 - ||v||^2
    vector[K] u     = Lam_n' * (y_n .* psi2_inv);
    vector[K] v     = mdivide_left_tri_low(L_M, u);
    real      quad_n = dot_self(y_n ./ psi_n) - dot_self(v);

    // Accumulate: -0.5 * (d*log(2*pi) + log|Sigma| + y' Sigma^{-1} y)
    target += -0.5 * (d * log(2 * pi()) + log_det_n + quad_n);
  }
}

generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
