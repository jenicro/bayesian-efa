// Bayesian EFA with MAR missing data via missingness-pattern
// marginalization. For each unique pattern of observed columns,
// the corresponding sub-vector is multivariate normal with mean
// 0 and the matching sub-block of Sigma = Lambda Omega Lambda' + diag(psi^2).
//
// Ragged data are stored flat with offset arrays.

data {
  int<lower=1> N;
  int<lower=2> P;
  int<lower=1> K;

  // free (unconstrained) loading entries
  int<lower=1> M;
  array[M] int<lower=1> row_idx;
  array[M] int<lower=1> col_idx;
  // positivity-constrained "anchor" loadings
  int<lower=0> n_diag;
  array[n_diag] int<lower=1> diag_row_idx;
  array[n_diag] int<lower=1> diag_col_idx;

  // pattern groups
  int<lower=1> n_patterns;
  array[n_patterns] int<lower=1> pat_dim;       // d_p
  array[n_patterns] int<lower=1> pat_nrows;     // rows in pattern
  int<lower=1> total_obs_idx;                   // sum d_p
  array[total_obs_idx] int<lower=1> obs_idx_flat;
  int<lower=1> total_y;                          // sum d_p * n_p
  array[total_y] real y_flat;

  real<lower=0> tau0;
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
  real<lower=0> lkj_eta;
}

parameters {
  vector<lower=0>[n_diag] diag_loadings;
  vector[M] z;
  vector<lower=0>[M] lam;
  real<lower=0> tau;
  real<lower=0> caux;
  vector<lower=0>[P] psi;
  cholesky_factor_corr[K] L_Omega;
}

transformed parameters {
  matrix[P, K] Lambda = rep_matrix(0, P, K);
  {
    real c = slab_scale * sqrt(caux);
    vector[M] lam_tilde =
      sqrt( (c^2 * square(lam)) ./ (c^2 + tau^2 * square(lam)) );
    vector[M] free_loadings = z .* (tau * lam_tilde);
    for (k in 1:n_diag) Lambda[diag_row_idx[k], diag_col_idx[k]] = diag_loadings[k];
    for (m in 1:M) Lambda[row_idx[m], col_idx[m]] = free_loadings[m];
  }
}

model {
  z   ~ std_normal();
  lam ~ student_t(3, 0, 1);
  tau ~ student_t(3, 0, tau0);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  diag_loadings ~ normal(0, 1);
  psi           ~ normal(0, 1);
  L_Omega       ~ lkj_corr_cholesky(lkj_eta);

  matrix[K, K] Omega_local = multiply_lower_tri_self_transpose(L_Omega);
  matrix[P, P] Sigma = Lambda * Omega_local * Lambda'
                       + diag_matrix(square(psi));
  for (p in 1:P) Sigma[p, p] += 1e-6;

  // walk through pattern groups
  int idx_off = 0;   // into obs_idx_flat
  int y_off   = 0;   // into y_flat
  for (g in 1:n_patterns) {
    int d  = pat_dim[g];
    int nr = pat_nrows[g];
    array[d] int cols;
    for (j in 1:d) cols[j] = obs_idx_flat[idx_off + j];

    matrix[d, d] Sub;
    for (a in 1:d) for (b in 1:d) Sub[a, b] = Sigma[cols[a], cols[b]];
    matrix[d, d] Lsub = cholesky_decompose(Sub);
    vector[d] mu0 = rep_vector(0, d);

    // Vectorise over all rows in this pattern group.
    array[nr] vector[d] Y_pat;
    for (r in 1:nr)
      for (j in 1:d)
        Y_pat[r][j] = y_flat[y_off + (r - 1) * d + j];
    Y_pat ~ multi_normal_cholesky(mu0, Lsub);

    idx_off += d;
    y_off   += d * nr;
  }
}

generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
