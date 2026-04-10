# Bayesian EFA Lab

A Streamlit research application for **Bayesian Exploratory Factor Analysis** with shrinkage priors, multiple inference backends, several identification strategies, and real-time recovery diagnostics against simulated ground truth.

---

## What it does

| Step | What you control |
|---|---|
| **1 · Data** | Simulate a factor structure (K, items/factor, loading range, cross-loadings, factor correlations, noise, missingness) **or** upload a CSV |
| **2 · Model** | Number of factors K to fit, LKJ concentration η, horseshoe slab scale, identification strategy |
| **3 · Backend** | PyMC ADVI (fast approximate), PyMC NUTS (full HMC), CmdStan NUTS (compiled, recommended) |
| **4 · Diagnose** | Posterior loadings Λ with credible intervals, factor correlations Ω, uniquenesses ψ, R̂ / ESS diagnostics, recovery vs. truth, factor-count diagnostics |

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 2. Install Python dependencies
pip install -r requirements.txt
pip install cmdstanpy               # Stan backend (optional but recommended)

# 3. Install CmdStan (once per machine)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# 4. Run
python -m streamlit run app.py
```

### Windows + rtools45

The Stan wrapper (`befa_stan.py`) prepends `C:\rtools45\usr\bin` and `C:\rtools45\x86_64-w64-mingw32.static.posix\bin` to `PATH` so that `cmdstanpy` can find `gcc` for compilation. If your toolchain lives elsewhere, edit lines 12–16 of `befa_stan.py`.

### Stan binaries

Compiled Stan binaries (`*.exe`, `*.hpp`) are **not** committed (they are platform-specific). On first use, `CmdStanModel(stan_file=...)` compiles them automatically and caches them beside the `.stan` files.

---

## The statistical model

### Factor analysis likelihood

EFA posits that P observed variables for subject i arise from K latent factors:

```
y_i = Λ f_i + ε_i

f_i ~ N(0, Ω)        factor scores, K-vector
ε_i ~ N(0, diag(ψ²)) item-specific noise, P-vector
```

Marginalising out the factor scores f_i analytically gives:

```
y_i ~ N(0, Σ)    where  Σ = Λ Ω Λ' + diag(ψ²)
```

This is the likelihood evaluated in all three Stan models and in the PyMC model. Factor scores are **never sampled** (except in the data-augmentation model for sparse missing data).

### Parameters

| Symbol | Meaning | Prior |
|---|---|---|
| **Λ** (P × K) | Loading matrix — how strongly each variable loads on each factor | Regularised horseshoe (free entries); HalfNormal(1) (anchor entries) |
| **Ω** (K × K) | Factor correlation matrix | LKJCorr(η) |
| **ψ** (P) | Uniqueness standard deviations (residual noise per variable) | HalfNormal(1) |

### Regularised horseshoe prior on loadings

Free loadings use the **regularised horseshoe** of Piironen & Vehtari (2017):

```
z_m   ~ N(0, 1)                        raw coefficient
λ_m   ~ HalfStudentT(3, 1)             local scale
τ     ~ HalfStudentT(3, τ₀)            global scale
c²    ~ InvGamma(slab_df/2, slab_df/2 · slab_scale²)   slab variance

λ̃_m  = sqrt( c² λ_m² / (c² + τ² λ_m²) )   regularised local scale
free loading = z_m · τ · λ̃_m
```

The horseshoe places most of its mass at 0 (sparse solution) while the slab allows large loadings to escape shrinkage. `τ₀ = (K / max(1, M-K)) / sqrt(N)` by default, giving the expected number of non-negligible loadings equal to K. You can override τ₀ in the UI.

Anchor / diagonal entries (needed for identification) use `HalfNormal(1)` — positivity-constrained, no global shrinkage.

### Factor correlations

`Ω ~ LKJCorr(η)`. The LKJ distribution (Lewandowski, Kurowicka, Joe 2009) is a prior over correlation matrices. `η = 1` is uniform over all valid correlation matrices; `η > 1` concentrates mass toward the identity (near-orthogonal factors); `η < 1` allows strong correlations. Default: `η = 2`.

---

## Identification strategies

The EFA likelihood is invariant to the transformation `(Λ, Ω) → (ΛA, A⁻¹ΩA⁻ᵀ)` for any invertible K×K matrix A. Without constraints, the posterior is a K²-dimensional ridge of equivalent solutions. The prior must break this rotational indeterminacy.

### 1 · Lower-triangular (Geweke-Zhou 1996)

The loading matrix is forced to be strictly lower-triangular with a positive diagonal:

```
Λ = [ λ₁₁   0    0   ]
    [ λ₂₁  λ₂₂   0   ]
    [ λ₃₁  λ₃₂  λ₃₃  ]
    [ λ₄₁  λ₄₂  λ₄₃  ]
    [ ...              ]
```

Items 0…K-1 are forced anchors: item j can only load on factors 0…j. This is fully Bayesian.

**Known pathology**: if item 0 does not naturally define factor 1 in the data, the horseshoe + ADVI can trap that factor at near-zero (the "empty factor" bug). Switch to `anchor` or `unconstrained + Promax`, or use NUTS.

### 2 · Anchor items (Aguilar & West 2000)

You supply K distinct anchor items — one per factor. Anchor k has a positive loading on factor k and structural zeros on factors k+1…K-1. All other items are fully free across all factors. Fully Bayesian.

**Good for**: cases where you know which items should define each factor (e.g., from theory or pilot data).

### 3 · Unconstrained + Promax (post-hoc rotation)

No structural constraints. All P×K loadings are free under the horseshoe. After sampling, a **shared-reference Promax** rotation is applied:

1. Compute the posterior-mean signal covariance **Σ̄ = mean(ΛΩΛᵀ)** — fully Bayesian.
2. Take its top-K eigen-decomposition → working basis **L_ref**.
3. Run **Promax** (oblique rotation, κ=4) on L_ref **once** → shared reference `(Λ*_ref, Ω*_ref, A_ref)`.
4. For every posterior draw: find orthogonal Procrustes `Q_d` aligning the draw to L_ref, then apply the composed oblique transform. This expresses all draws in the **same** simple-structure basis.

Promax (Hendrickson & White 1964) first applies varimax, then a least-squares fit to the sparse target `sign(Λ)|Λ|^κ`. It is oblique (allows correlated factors), consistent with `Ω ~ LKJCorr`.

**Honest accounting**: the posterior over **Σ = ΛΩΛᵀ is fully Bayesian**. The posterior over Λ after Promax is a plug-in point estimate via a deterministic function of the posterior mean. Credible intervals on loadings measure the spread of aligned draws around the reference, not honest posterior uncertainty. Point estimates are reliable; treat width with caution.

**Good for**: exploratory work where you want clean simple structure and are not making formal claims about loading uncertainty.

---

## Missing data

All models assume **MAR (Missing At Random)**. Three Stan implementations are available (select in the UI when missingness > 0 and CmdStan backend is chosen):

### Woodbury FIML (default, recommended)

Marginalises factor scores analytically using the **Woodbury matrix identity**. For each observed row n with observed column set Sₙ:

```
y_{n,Sₙ} ~ N(0, Λ_{Sₙ} Ω Λ_{Sₙ}' + diag(ψ_{Sₙ}²))
```

Key identities (all K×K — no d×d Cholesky):
```
log|Σ_S| = log|D_S| + log|Ω| + log|M_S|
Σ_S⁻¹   = D_S⁻¹ - D_S⁻¹ Λ_S M_S⁻¹ Λ_S' D_S⁻¹
y' Σ_S⁻¹ y = ||y/ψ_S||² - ||L_M⁻¹ u||²

where M_S = Ω⁻¹ + Λ_S' D_S⁻¹ Λ_S  (K×K only)
      u   = Λ_S' (y / ψ_S²)
```

Cost per HMC step: **O(n_obs · K² + N · K³)** — independent of missingness pattern complexity. Mathematically identical to pattern-mixture FIML.

### Pattern-mixture FIML (`befa_missing.stan`)

Groups rows by unique missingness pattern. One d×d Cholesky per group per HMC step. Cost: **O(n_patterns · d³)**. Practical only when missingness is low (< 30%) or P is small.

### Data augmentation (`befa_augmented.stan`)

Samples factor scores F explicitly as N×K extra parameters. Likelihood reduces to one univariate normal per observed cell: **O(n_obs)** per step — no Cholesky at all. Useful when N×K is manageable and missingness is extreme (> 80%). Can have mixing issues with the horseshoe prior because the F × Λ likelihood is invariant to many joint transformations.

---

## Output tabs

| Tab | Contents |
|---|---|
| **Loadings Λ** | Posterior-mean heatmap + table with mean, SD, and credible interval per loading. **✓** in *excl 0* = CI entirely above or below zero |
| **Factor corr Ω** | Heatmap of posterior-mean correlations + CI table for all off-diagonal pairs |
| **Uniquenesses ψ** | Bar chart (CI error bars) + table. High ψ = variable not well explained by the factors |
| **Sampling diag** | ELBO trace (ADVI) or R̂ / ESS table (NUTS). R̂ < 1.01, ESS > 400 per chain is the target |
| **Recovery** | Procrustes-aligned comparison to simulation truth (requires simulated data). Shows RMSE, max error |
| **K diagnostic** | Parallel analysis, Kaiser criterion, posterior SSL, and shrinkage-based active-factor count |
| **Save** | Download posterior as ArviZ NetCDF (PyMC) or long-format CSV (any backend) |

### Interpreting credible intervals

A **90% credible interval** [lo, hi] means: given the model and data, 90% of the posterior probability mass for that parameter lies between lo and hi. This is the direct Bayesian statement of uncertainty — unlike a frequentist confidence interval, it refers to the parameter itself, not to hypothetical repeated samples.

The **✓ flag** (`excl 0`) marks parameters whose entire 90% CI is strictly positive or strictly negative. For a loading, this means the data provide strong evidence that the variable loads on that factor in the stated direction, even accounting for horseshoe shrinkage. For a factor correlation, it means the two factors are credibly correlated.

---

## Factor-count diagnostics

Four methods are shown side by side:

| Method | Idea | Notes |
|---|---|---|
| **Parallel analysis** (Horn 1965) | Compare observed eigenvalues to those from random data (permuted Y). K = count above the 95th percentile null | Works on the raw data correlation matrix; does not use the Bayesian fit |
| **Kaiser criterion** | K = count of eigenvalues > 1 | Tends to over-extract; shown for reference |
| **Posterior SSL** | K = count of columns with posterior-mean SSL > 1 | SSL = sum of squared loadings for a column. Simple but ignores uncertainty |
| **Shrinkage factor count** | For each draw, compute SSL per column. Column is "active" if SSL > 1. K_active = count. Returns full posterior distribution over K | The principled horseshoe diagnostic. Shows P(K=k) and per-column P(active) |

---

## Stan console

Stan's iteration output goes to the **terminal** where Streamlit was launched (via `show_console=True` — Stan writes to file descriptor 1 directly, bypassing Python's `sys.stdout`). The Streamlit console panel captures Python-level log messages from the wrapper and polls the chain CSV files every 2 seconds to display a draw counter.

---

## File reference

```
├── app.py                  Streamlit UI (single source of truth for all UI logic)
├── bayesian_efa.py         PyMC model: build_model, fit_fast (ADVI), fit_full (NUTS)
├── befa.stan               Stan: dense data, no missingness
├── befa_missing.stan       Stan: pattern-mixture FIML
├── befa_woodbury.stan      Stan: Woodbury-identity FIML (recommended for missing data)
├── befa_augmented.stan     Stan: data-augmentation (explicit factor scores, sparse data)
├── befa_stan.py            cmdstanpy wrapper: data prep, model selection, fit_stan()
├── fit_runner.py           Threaded fit runner: progress, cancel, Stan console polling
├── simulator.py            Factor structure simulator
├── diagnostics.py          Parallel analysis, SSL, shrinkage factor-count
├── post_process.py         Varimax, Promax, shared-reference posterior rotation
├── test_recovery.py        Smoke test: simulate → fit → check recovery
├── test_stan.py            Smoke test: Stan backend only
└── requirements.txt
```

---

## References

- **Piironen & Vehtari (2017)** — Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics.*
- **Lewandowski, Kurowicka & Joe (2009)** — Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis.* (LKJ prior)
- **Geweke & Zhou (1996)** — Measuring the pricing error of the arbitrage pricing theory. *Review of Financial Studies.* (lower-triangular identification)
- **Aguilar & West (2000)** — Bayesian dynamic factor models and variance matrix discounting. *JASA.* (anchor item identification)
- **Horn (1965)** — A rationale and test for the number of factors in factor analysis. *Psychometrika.* (parallel analysis)
- **Hendrickson & White (1964)** — Promax: A quick method for rotation to oblique simple structure. *British Journal of Statistical Psychology.*
- **Woodbury (1950)** — Inverting modified matrices. *Memorandum Rept. 42, Statistical Research Group, Princeton.*
