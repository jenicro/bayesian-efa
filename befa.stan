// Bayesian Exploratory Factor Analysis with correlated factors and
// regularized-horseshoe shrinkage on the cross-loadings.
//
// Identifiability: Lambda is lower-triangular with positive diagonal
// (Geweke & Zhou 1996). Factor variances fixed to 1, factor
// correlation matrix Omega ~ LKJ(eta).
//
// Latent factors are marginalized analytically:
//   Y_i ~ MvNormal(0, Lambda Omega Lambda^T + diag(psi^2))
//
// Shrinkage prior: regularized horseshoe (Piironen & Vehtari 2017)
// on the strict-lower-triangular free entries of Lambda.

data {
  int<lower=1> N;            // observations
  int<lower=2> P;            // variables
  int<lower=1> K;            // factors (K < P)
  int<lower=1> M;            // number of free (unconstrained) loadings
  array[M] int<lower=1> row_idx;
  array[M] int<lower=1> col_idx;
  int<lower=0> n_diag;       // number of positivity-constrained "anchor" loadings
  array[n_diag] int<lower=1> diag_row_idx;
  array[n_diag] int<lower=1> diag_col_idx;
  array[N] vector[P] Y;      // data (assumed standardized)
  real<lower=0> tau0;        // global shrinkage scale
  real<lower=0> slab_scale;  // regularized-horseshoe slab
  real<lower=0> slab_df;
  real<lower=0> lkj_eta;
}

parameters {
  vector<lower=0>[n_diag] diag_loadings;   // positivity-constrained anchors
  vector[M] z;                             // standard-normal aux for loadings
  vector<lower=0>[M] lam;                  // local shrinkage
  real<lower=0> tau;                       // global shrinkage
  real<lower=0> caux;                      // slab scale aux (inv-gamma)
  vector<lower=0>[P] psi;                  // uniquenesses (sd)
  cholesky_factor_corr[K] L_Omega;         // factor correlation chol
}

transformed parameters {
  matrix[P, K] Lambda = rep_matrix(0, P, K);
  vector[M] free_loadings;
  {
    real c = slab_scale * sqrt(caux);
    vector[M] lam_tilde =
      sqrt( (c^2 * square(lam)) ./ (c^2 + tau^2 * square(lam)) );
    free_loadings = z .* (tau * lam_tilde);
  }
  for (k in 1:n_diag)
    Lambda[diag_row_idx[k], diag_col_idx[k]] = diag_loadings[k];
  for (m in 1:M)
    Lambda[row_idx[m], col_idx[m]] = free_loadings[m];
}

model {
  // ---- Regularized horseshoe (Piironen & Vehtari) ------------------
  z   ~ std_normal();
  lam ~ student_t(3, 0, 1);
  tau ~ student_t(3, 0, tau0);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);

  // ---- Other priors -----------------------------------------------
  diag_loadings ~ normal(0, 1);    // half-normal via <lower=0>
  psi           ~ normal(0, 1);    // half-normal
  L_Omega       ~ lkj_corr_cholesky(lkj_eta);

  // ---- Likelihood (factors marginalized) --------------------------
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
  matrix[P, P] Sigma = Lambda * Omega * Lambda'
                       + diag_matrix(square(psi));
  // jitter for numerical PD
  for (p in 1:P) Sigma[p, p] += 1e-6;
  matrix[P, P] L_Sigma = cholesky_decompose(Sigma);
  // Vectorised over all N rows — much faster than a per-row loop,
  // especially during initialisation retries for large K.
  Y ~ multi_normal_cholesky(rep_vector(0, P), L_Sigma);
}

generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
