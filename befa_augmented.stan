// Bayesian EFA for sparse / highly-missing data via explicit factor scores
// (data-augmentation approach).
//
// WHY THIS MODEL EXISTS
// ---------------------
// The analytic FIML model (befa_missing.stan) marginalises factor scores and
// evaluates one sub-block Cholesky per unique missingness pattern per step.
// With P > 50 and missingness > 40 % the number of unique patterns approaches
// N, making each log-prob evaluation O(N * d^3) where d ~ P*(1-miss).
//
// Here we instead *sample* the factor scores F explicitly.  The likelihood
// reduces to one univariate normal per observed cell: O(n_obs) per step.
// The cost is N*K extra parameters, but for 80 % missingness that is a
// net win by one to two orders of magnitude.
//
// MODEL
// -----
//   F[n,:] ~ MVN(0, Omega)           (latent factor scores)
//   y[n,p] | F[n], Lambda, psi
//           ~ Normal(Lambda[p,:] . F[n,:], psi[p])   only for observed cells
//
// Factor scores use a non-centred reparameterisation to improve sampler
// geometry: F_raw[n,:] ~ N(0,I), then F = F_raw * chol(Omega)'.
//
// IMPORTANT: initialise F_raw with non-zero values (e.g. 0.5*N(0,I)).
// The likelihood gradient w.r.t. z (loadings) is proportional to F[n,k].
// If F_raw=0 at init, all loading gradients are 0 and loadings collapse
// into the horseshoe spike.  See _make_stan_init in befa_stan.py.
//
// Loading identification and horseshoe prior are identical to befa.stan.

data {
  int<lower=1> N;
  int<lower=2> P;
  int<lower=1> K;

  // Sparse observation format (only observed cells)
  int<lower=1>                        n_obs;
  array[n_obs] int<lower=1, upper=N> obs_row;
  array[n_obs] int<lower=1, upper=P> obs_col;
  vector[n_obs]                       y_vals;

  // Loading structure (same convention as befa.stan)
  int<lower=1>                        M;
  array[M]     int<lower=1>           row_idx;
  array[M]     int<lower=1>           col_idx;
  int<lower=0>                        n_diag;
  array[n_diag] int<lower=1>          diag_row_idx;
  array[n_diag] int<lower=1>          diag_col_idx;

  // Horseshoe / LKJ hyperparameters
  real<lower=0> tau0;
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
  real<lower=0> lkj_eta;
}

parameters {
  // ---- Loading-matrix components (same as befa.stan) ------------------
  vector<lower=0>[n_diag] diag_loadings;
  vector[M]               z;
  vector<lower=0>[M]      lam;
  real<lower=0>           tau;
  real<lower=0>           caux;

  // ---- Uniquenesses ----------------------------------------------------
  vector<lower=0>[P]      psi;

  // ---- Factor correlation Cholesky ------------------------------------
  cholesky_factor_corr[K] L_Omega;

  // ---- Factor scores (non-centred) ------------------------------------
  // F_raw[n,:] ~ N(0, I_K).  Transformed to F = F_raw * L_Omega' so that
  // each row F[n,:] ~ MVN(0, Omega).
  matrix[N, K]            F_raw;
}

transformed parameters {
  // ---- Assemble loading matrix ----------------------------------------
  matrix[P, K] Lambda = rep_matrix(0, P, K);
  {
    real       c          = slab_scale * sqrt(caux);
    vector[M]  lam_tilde  =
      sqrt( (c^2 * square(lam)) ./ (c^2 + tau^2 * square(lam)) );
    vector[M]  free_loadings = z .* (tau * lam_tilde);
    for (k in 1:n_diag)
      Lambda[diag_row_idx[k], diag_col_idx[k]] = diag_loadings[k];
    for (m in 1:M)
      Lambda[row_idx[m], col_idx[m]] = free_loadings[m];
  }

  // ---- Non-centred factor scores --------------------------------------
  // Each row of F_raw is iid N(0, I); multiplying by L_Omega' gives
  // rows of F distributed as MVN(0, Omega).
  matrix[N, K] F = F_raw * L_Omega';
}

model {
  // ---- Regularised horseshoe (Piironen & Vehtari 2017) ---------------
  z    ~ std_normal();
  lam  ~ student_t(3, 0, 1);
  tau  ~ student_t(3, 0, tau0);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);

  // ---- Other priors ---------------------------------------------------
  diag_loadings ~ normal(0, 1);
  psi           ~ normal(0, 1);
  L_Omega       ~ lkj_corr_cholesky(lkj_eta);

  // ---- Factor score prior (non-centred) -------------------------------
  to_vector(F_raw) ~ std_normal();

  // ---- Likelihood: one univariate normal per observed cell ------------
  // mean = Lambda[p,:] · F[n,:],  sd = psi[p]
  // This is O(n_obs) regardless of missingness pattern complexity.
  for (obs in 1:n_obs) {
    int n = obs_row[obs];
    int p = obs_col[obs];
    y_vals[obs] ~ normal(Lambda[p] * F[n]', psi[p]);
  }
}

generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
