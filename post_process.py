"""Post-hoc identification for Bayesian factor models.

When the sampler is run with `identification='unconstrained'` (no
triangular or anchor constraints), the posterior is invariant to
rotation, column permutation, and column sign-flip of (Lambda, Omega).
We resolve these indeterminacies as a post-processing step:

1. **Promax rotation** (Hendrickson & White 1964) — an *oblique*
   rotation: varimax first, then a least-squares fit to a power target
   ``sign(Λ) · |Λ|^κ``. This produces a simple-structure pattern
   matrix while allowing the factor correlation matrix Φ to be
   non-identity, which is the correct behavior for our LKJ-prior
   correlated-factor model. Varimax alone would force Φ = I, which is
   *wrong* for a model that has a factor correlation matrix as a
   parameter.
2. **Column reordering** by sum-of-squared-loadings (largest first).
3. **Sign flip** so that the largest-magnitude entry in each column is
   positive.

Because the unconstrained model has (Λ, Ω) with a generally non-trivial
Ω, we first *whiten* each draw (Λ_u = Λ · chol(Ω)) so that the working
loading matrix has orthogonal factors, apply Promax, and read off both
the rotated pattern matrix and the induced factor correlation matrix.
The full transform is invariant: Λ* Φ* Λ*^T = Λ Ω Λ^T. This is cheap
per draw (a few K×K decompositions).
"""
from __future__ import annotations

import numpy as np


def varimax(Lambda: np.ndarray, gamma: float = 1.0, max_iter: int = 100,
            tol: float = 1e-6) -> np.ndarray:
    """Kaiser's varimax rotation. Returns the K×K rotation matrix R such
    that Lambda @ R is the rotated loading matrix. `gamma=1` is
    standard varimax; `gamma=0` is quartimax."""
    P, K = Lambda.shape
    if K < 2:
        return np.eye(K)
    R = np.eye(K)
    d = 0.0
    for _ in range(max_iter):
        L_rot = Lambda @ R
        # column-mean of squared loadings
        col_mean_sq = (L_rot ** 2).mean(axis=0) if gamma > 0 else 0.0
        # gradient step: SVD of Lambda^T (L_rot^3 - gamma * L_rot * col_mean_sq)
        B = L_rot ** 3 - gamma * L_rot * col_mean_sq
        U, s, Vt = np.linalg.svd(Lambda.T @ B, full_matrices=False)
        R = U @ Vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return R


def _reorder_and_flip(L: np.ndarray, O: np.ndarray | None = None):
    """Reorder columns of L by descending SSL, flip signs so the
    largest-magnitude entry in each column is positive. Apply matching
    permutation + sign-flip to O if provided."""
    K = L.shape[1]
    ssl = (L ** 2).sum(axis=0)
    order = np.argsort(-ssl)
    L = L[:, order]
    signs = np.ones(K)
    for k in range(K):
        j_max = int(np.argmax(np.abs(L[:, k])))
        if L[j_max, k] < 0:
            signs[k] = -1.0
    L = L * signs
    if O is not None:
        O = O[np.ix_(order, order)] * np.outer(signs, signs)
    return L, O, order, signs


def promax(
    Lambda_u: np.ndarray,
    kappa: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Promax oblique rotation of an *orthogonal* loading matrix.

    Takes Λ_u with implicit Φ = I (i.e., ``Λ_u Λ_u^T`` is the model
    covariance) and returns ``(Λ*, Φ*)`` such that
    ``Λ* Φ* Λ*^T == Λ_u Λ_u^T`` and Φ* is a correlation matrix. Φ* is
    generally non-identity (oblique rotation).
    """
    P, K = Lambda_u.shape
    if K < 2:
        return Lambda_u.copy(), np.eye(K)

    # Step 1: varimax orthogonal rotation
    R = varimax(Lambda_u)
    V = Lambda_u @ R                            # (P, K)

    # Step 2: Kaiser row-normalization (standard for Promax target)
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    Vn = V / row_norms

    # Step 3: power target and least-squares Procrustes
    T = np.sign(Vn) * np.abs(Vn) ** kappa
    # Q minimizes ||Vn @ Q - T||_F
    Q, *_ = np.linalg.lstsq(Vn, T, rcond=None)  # (K, K)

    # Step 4: compose with varimax → full oblique transform A = R @ Q
    A = R @ Q
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    B = A_inv @ A_inv.T                         # unnormalized Φ
    d = np.sqrt(np.clip(np.diag(B), 1e-12, None))
    D_inv = np.diag(1.0 / d)
    Phi = D_inv @ B @ D_inv                     # unit-diagonal Φ*

    # Rescale loadings so the overall model covariance is unchanged
    Lambda_star = (Lambda_u @ A) * d            # columns scaled by d
    return Lambda_star, Phi


def varimax_align_posterior(
    Lambda_post: np.ndarray,
    Omega_post: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Varimax-align a posterior from an orthogonal-factors model (Ω = I).

    Uses a single shared reference rotation computed from the posterior-mean
    covariance — same philosophy as promax_align_posterior, but orthogonal
    rotation only (correct when Ω is fixed to identity).

    Returns Lambda_aligned (same shape) and Omega_aligned (identity, same shape).
    """
    L = np.asarray(Lambda_post)
    C, D, P, K = L.shape

    # Posterior-mean covariance (Omega=I so Sigma_F = Lambda Lambda^T)
    Sigma_bar = np.zeros((P, P))
    for c in range(C):
        for d in range(D):
            Sigma_bar += L[c, d] @ L[c, d].T
    Sigma_bar /= (C * D)

    # Top-K eigendecomposition → orthogonal reference loadings
    eigvals, eigvecs = np.linalg.eigh(Sigma_bar)
    order = np.argsort(-eigvals)[:K]
    eigvals_k = np.clip(eigvals[order], 1e-10, None)
    eigvecs_k = eigvecs[:, order]
    Lu_ref = eigvecs_k * np.sqrt(eigvals_k)  # (P, K)

    # Varimax rotate reference once to get the shared basis
    R_ref = varimax(Lu_ref)
    Lambda_ref = Lu_ref @ R_ref

    # Reorder columns by SSL, sign-flip
    Lambda_ref, _, perm_ref, signs_ref = _reorder_and_flip(Lambda_ref)
    # Apply same permutation+signs to R_ref
    R_ref = R_ref[:, perm_ref] * signs_ref[np.newaxis, :]

    # Align every draw to the shared reference via orthogonal Procrustes
    L_out = np.empty_like(L)
    O_out = np.broadcast_to(np.eye(K), (C, D, K, K)).copy()
    for c in range(C):
        for d in range(D):
            Ld = L[c, d]
            # Orthogonal Procrustes: find Q s.t. Ld @ Q ≈ Lambda_ref
            U_, _s, Vt = np.linalg.svd(Ld.T @ Lambda_ref, full_matrices=False)
            Q_d = U_ @ Vt
            L_out[c, d] = Ld @ Q_d
    return L_out, O_out


def _chol_psd(M: np.ndarray) -> np.ndarray:
    K = M.shape[0]
    try:
        return np.linalg.cholesky(M + 1e-10 * np.eye(K))
    except np.linalg.LinAlgError:
        w, V_ = np.linalg.eigh(M)
        w = np.clip(w, 1e-10, None)
        return V_ * np.sqrt(w)


def promax_align_posterior(
    Lambda_post: np.ndarray,
    Omega_post: np.ndarray,
    kappa: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Promax-align a posterior of unconstrained factor loadings.

    The rotation is computed **once** from the posterior-mean implied
    covariance Σ̄ = E[Λ Ω Λᵀ], then applied to every draw. This is the
    correct way to post-process an unconstrained Bayesian factor fit:
    per-draw rotation (which I tried first) leads to rotation-basis
    drift across draws and muddies the averaged loading matrix with
    phantom cross-loadings. Computing the rotation from the posterior
    *mean covariance* produces a single shared basis that every draw
    is expressed in, so the posterior mean of the aligned loadings is
    the genuine simple-structure solution.

    Algorithm (per-draw, with a shared reference rotation)
    -------------------------------------------------------
    1. Compute Σ̄ by averaging ``Λ Ω Λᵀ`` across draws.
    2. Get ``Λ_ref`` via eigendecomposition of Σ̄ keeping the top K
       components — this is the MLE simple-structure working basis.
    3. Promax-rotate ``Λ_ref`` once to get a reference oblique basis
       (``Λ*_ref``, ``Φ*_ref``), then reorder by SSL and sign-flip.
    4. For every draw, solve for the unique oblique transform A_d that
       maps that draw's orthogonal whitening ``Λ_u = Λ · chol(Ω)`` onto
       the reference basis and apply the same transform.

    Returns
    -------
    Lambda_aligned, Omega_aligned — same shapes as inputs. Ω is a proper
    correlation matrix (unit diagonal) for every draw.
    """
    L = np.asarray(Lambda_post)
    O = np.asarray(Omega_post)
    C, D, P, K = L.shape

    # ---- Step 1-2: posterior-mean reference working basis ------------
    Sigma_bar = np.zeros((P, P))
    for c in range(C):
        for d in range(D):
            Sigma_bar += L[c, d] @ O[c, d] @ L[c, d].T
    Sigma_bar /= (C * D)
    # Top-K eigendecomposition → orthogonal reference loadings
    eigvals, eigvecs = np.linalg.eigh(Sigma_bar)
    order = np.argsort(-eigvals)[:K]
    eigvals_k = np.clip(eigvals[order], 1e-10, None)
    eigvecs_k = eigvecs[:, order]
    Lu_ref = eigvecs_k * np.sqrt(eigvals_k)       # (P, K), orthogonal basis

    # ---- Step 3: Promax rotate the reference once -------------------
    # promax() returns Λ* and Φ* such that Λ* Φ* Λ*ᵀ == Lu_ref Lu_refᵀ.
    # We also want the explicit oblique transform A such that
    #   Λ* = Lu_ref @ A_ref           with Φ* = (A_ref⁻ᵀ A_ref⁻¹) rescaled
    # so we can reuse it to align every draw in the same basis.
    # Recompute it step-by-step here to keep A_ref in hand.
    R = varimax(Lu_ref)
    V = Lu_ref @ R
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    Vn = V / row_norms
    T = np.sign(Vn) * np.abs(Vn) ** kappa
    Q, *_ = np.linalg.lstsq(Vn, T, rcond=None)
    A_unscaled = R @ Q                             # (K, K)
    A_inv = np.linalg.pinv(A_unscaled)
    B = A_inv @ A_inv.T
    d_ref = np.sqrt(np.clip(np.diag(B), 1e-12, None))
    # Rescaled oblique transform: rotation that produces Λ* with unit-Φ diag
    A_ref = A_unscaled * d_ref[np.newaxis, :]      # (K, K)
    Lambda_ref = Lu_ref @ A_ref

    # Reorder columns by SSL, then sign-flip so the largest-magnitude
    # entry in each column is positive. Apply the same column
    # permutation+signs to A_ref so the shared transform matches.
    ssl_ref = (Lambda_ref ** 2).sum(axis=0)
    perm = np.argsort(-ssl_ref)
    Lambda_ref = Lambda_ref[:, perm]
    A_ref = A_ref[:, perm]
    signs_ref = np.ones(K)
    for k in range(K):
        if Lambda_ref[np.argmax(np.abs(Lambda_ref[:, k])), k] < 0:
            signs_ref[k] = -1.0
    Lambda_ref = Lambda_ref * signs_ref
    A_ref = A_ref * signs_ref[np.newaxis, :]

    # ---- Step 4: express every draw in the shared reference basis ---
    # Each draw's orthogonal whitening Lu_d = Λ_d · chol(Ω_d). We want
    # to map Lu_d onto Lu_ref (same factor basis), then apply A_ref.
    # The mapping from Lu_d to Lu_ref in the K-dimensional factor space
    # is an orthogonal Procrustes rotation:
    #   minimize ||Lu_d @ Q_d - Lu_ref||   s.t. Q_dᵀQ_d = I
    #   → Q_d = U Vᵀ where (Lu_dᵀ Lu_ref) = U S Vᵀ
    L_out = np.empty_like(L)
    O_out = np.empty_like(O)
    for c in range(C):
        for d in range(D):
            Ld = L[c, d]
            Od = O[c, d]
            chol_d = _chol_psd(Od)
            Lu_d = Ld @ chol_d
            U_, _s, Vt = np.linalg.svd(Lu_d.T @ Lu_ref, full_matrices=False)
            Q_d = U_ @ Vt                          # (K, K) orthogonal
            # Shared oblique transform on this draw:
            T_d = chol_d @ Q_d @ A_ref             # (K, K)
            L_aligned = Ld @ T_d
            # Induced factor correlation matrix from this draw's Ω
            # under the transform T_d:  Φ_d = T_d⁻¹ Ω_d T_d⁻ᵀ
            try:
                T_inv = np.linalg.inv(T_d)
            except np.linalg.LinAlgError:
                T_inv = np.linalg.pinv(T_d)
            Phi_d = T_inv @ Od @ T_inv.T
            # Rescale to a proper correlation matrix (unit diagonal),
            # absorbing the scale into the loadings so Λ Φ Λᵀ is unchanged.
            dd = np.sqrt(np.clip(np.diag(Phi_d), 1e-12, None))
            Phi_d = Phi_d / np.outer(dd, dd)
            L_aligned = L_aligned * dd[np.newaxis, :]
            L_out[c, d] = L_aligned
            O_out[c, d] = Phi_d
    return L_out, O_out
