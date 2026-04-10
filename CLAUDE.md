# CLAUDE.md — Fun-With-Bayes-EFA

A Streamlit "lab" for **Bayesian Exploratory Factor Analysis** with shrinkage priors, three inference backends, multiple identification strategies, missing-data FIML, and recovery diagnostics against simulated ground truth. This file is the single source of truth for anyone (human or LLM) picking the project up mid-stream. Read it end to end before editing code.

---

## 1. What the project is

A single-user research/teaching app that lets you:

1. **Design** a factor structure (K, items-per-factor, loading magnitudes, cross-loading probability, factor correlations, noise, missingness) **or** upload a CSV.
2. **Fit** a Bayesian EFA model with one of three backends:
   - PyMC **ADVI** (mean-field variational, fast)
   - PyMC **NUTS** (full HMC posterior)
   - **CmdStan NUTS** (compiled Stan, typically the most reliable for this model)
3. **Choose an identification strategy** for the rotational indeterminacy of Λ:
   - **Lower-triangular** (Geweke-Zhou 1996)
   - **Anchor items** (generalised Geweke-Zhou, Aguilar & West 2000)
   - **Unconstrained + Promax post-hoc** (oblique simple-structure rotation of the posterior)
4. **Choose a missing-data model** (CmdStan only):
   - **Woodbury FIML** — marginalised, K×K Cholesky only, recommended
   - **Pattern-mixture FIML** — original d×d Cholesky per pattern
   - **Data augmentation** — explicit factor scores, O(n_obs) univariate normals
5. **Diagnose** the fit: posterior loadings Λ with credible intervals, factor correlations Ω, uniquenesses ψ, sampling diagnostics (R̂ / ESS / ELBO), recovery vs. truth (Procrustes-aligned), and **number-of-factors diagnostics** (parallel analysis, Kaiser, posterior SSL, shrinkage-based active-factor count).
6. **Save** posteriors as NetCDF / pickle / long CSV.

---

## 2. Repository layout

```
Fun-With-Bayes-EFA/
├── app.py                  Streamlit front-end (single source of UI truth)
├── bayesian_efa.py         PyMC model builder + fit_fast / fit_full
├── befa.stan               CmdStan model — dense Y, no missingness
├── befa_missing.stan       CmdStan model — pattern-mixture FIML
├── befa_woodbury.stan      CmdStan model — Woodbury-identity FIML (recommended)
├── befa_augmented.stan     CmdStan model — data-augmentation (explicit factor scores)
├── befa_stan.py            cmdstanpy wrapper (data prep, model selection, fit)
├── fit_runner.py           Threaded fit runner with live progress + cancel
├── simulator.py            Factor-structure simulator
├── diagnostics.py          Parallel analysis, SSL, shrinkage factor-count diagnostic
├── post_process.py         Varimax / Promax + shared-reference rotation alignment
├── test_recovery.py        Smoke test — simulate → PyMC fit → check recovery
├── test_stan.py            Smoke test — CmdStan only
├── requirements.txt        pymc, arviz, numpy, pandas, matplotlib, streamlit
├── README.md               User-facing docs with statistical explanations
└── CLAUDE.md               THIS FILE
```

Stan compiled binaries (`*.exe`, `*.hpp`) are gitignored — they are platform-specific and recompiled automatically by `CmdStanModel(stan_file=...)`.

---

## 3. How to run it

```bash
pip install -r requirements.txt
pip install cmdstanpy                               # Stan backend
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"  # once per machine
python -m streamlit run app.py
```

**Windows rtools45**: `befa_stan.py` lines 12–16 prepend `C:\rtools45\usr\bin` and `C:\rtools45\x86_64-w64-mingw32.static.posix\bin` to PATH. Edit those paths if your toolchain lives elsewhere.

**Stan console**: Stan's C binary writes iteration output directly to file descriptor 1 (the real terminal), bypassing Python's `sys.stdout`. The Streamlit console panel captures Python-level log messages from the wrapper and polls the chain CSV files every 2 s to display a draw counter. Full iteration output always appears in the terminal where `streamlit run` was launched.

---

## 4. The model (one definition, implemented in four Stan variants + PyMC)

### Likelihood (factors marginalised analytically)

```
y_i | Λ, Ω, ψ  ~  N(0, Λ Ω Λᵀ + diag(ψ²))            i = 1..N
```

Data Y is standardised (column-wise, NaN-safe) before entering the likelihood. Factor scores f_i are **not** sampled in the primary models — they are analytically integrated out.

### Priors

- **Free loadings** — regularised horseshoe (Piironen & Vehtari 2017):
  - `λ_m ~ HalfStudentT(3, 1)` local scale
  - `τ   ~ HalfStudentT(3, τ₀)` global scale; `τ₀ = (K / max(1, M-K)) · (1/√N)` heuristic
  - `c²  ~ InvGamma(slab_df/2, slab_df/2 · slab_scale²)` slab
  - free loading = `z_m · τ · λ̃_m` where `λ̃_m = sqrt(c²λ_m² / (c² + τ²λ_m²))`
  - `HalfStudentT(ν=3)` instead of `HalfCauchy` — better-behaved with ADVI
- **Anchor / diagonal loadings**: `HalfNormal(1)` — positivity-constrained
- **Factor correlation**: `Ω ~ LKJCorr(η)` — K×K correlation matrix, factor variances fixed at 1
- **Uniquenesses**: `ψ ~ HalfNormal(1)`

### Identification strategies (see `_resolve_identification` in `bayesian_efa.py`)

The likelihood is invariant to `(Λ, Ω) → (ΛA, A⁻¹ΩA⁻ᵀ)`. The prior must break this.

1. **`lower_triangular`** — Geweke-Zhou 1996. Strict lower-triangular Λ with positive diagonal. Implemented as anchors = items `0..K-1`. **Known pathology**: if item 0 doesn't naturally define factor 1, horseshoe + ADVI can trap a real factor at ≈ 0 (the "dead column" bug). See §8.

2. **`anchor`** — Generalised GZ. User supplies K distinct anchor indices. Anchor k has positive loading on factor k and structural zeros on factors k+1…K-1. Fully Bayesian.

3. **`unconstrained`** — No structural constraints. All P·K entries are free. Post-hoc shared-reference Promax rotation. **NOT fully Bayesian on Λ** — see §5.

`lower_triangular` routes through `anchor` internally with anchors = `[0..K-1]`.

---

## 5. Post-hoc identification: shared-reference Promax

Lives in `post_process.py`. Used only when `identification == "unconstrained"`.

**Why Promax**: Varimax is orthogonal (Φ = I). Our model has `Ω ~ LKJ`, so oblique rotation is correct. Promax (Hendrickson & White 1964): varimax first, then least-squares fit to `sign(Λ)|Λ|^κ` (default κ=4).

**Why shared reference**: per-draw Promax converges to slightly different bases → averaging smears phantom cross-loadings. Instead:
1. Compute `Σ̄ = mean(Λ_d Ω_d Λ_dᵀ)` — posterior-mean signal covariance.
2. Top-K eigen-decomposition → working basis `Lu_ref`.
3. Run Promax on `Lu_ref` **once** → shared `(Λ*_ref, Ω*_ref, A_ref)`.
4. For every draw: find orthogonal Procrustes `Q_d` (SVD of `Lu_d' Lu_ref`), apply composed transform `T_d = chol(Ω_d) · Q_d · A_ref`.

**Honest accounting**: the posterior over `Σ_F = ΛΩΛᵀ` is fully Bayesian. The posterior over Λ after Promax is a plug-in point estimate. Credible intervals measure spread of aligned draws, not honest posterior uncertainty. Point estimates are fine; CIs should be interpreted with caution.

---

## 6. Missing-data handling

### Woodbury FIML (`befa_woodbury.stan`) — default for missing data

Marginalises factor scores analytically. For each observed row n with observed set Sₙ:
```
y_{n,Sₙ} ~ N(0, Λ_{Sₙ} Ω Λ_{Sₙ}' + D_{Sₙ})
```
Using the Woodbury identity and matrix determinant lemma (see §4 of README for full derivation), the log-likelihood decomposes into K×K operations only:
```
M_S = Ω⁻¹ + Λ_S' D_S⁻¹ Λ_S    (K×K)
log|Σ_S| = log|D_S| + log|Ω| + log|M_S|
y' Σ_S⁻¹ y = ||y/ψ_S||² - ||L_M⁻¹ u||²   where u = Λ_S'(y/ψ_S²)
```
Cost: **O(n_obs·K² + N·K³)** per HMC step. 50–100× faster than pattern-mixture for high missingness. Mathematically identical result.

**Implementation notes**:
- `M_raw` is symmetrised as `0.5 * (M_raw + M_raw')` before `cholesky_decompose` to prevent floating-point asymmetry errors at extreme proposals during warmup.
- `Ω⁻¹` is computed as `crossprod(mdivide_left_tri_low(L_Omega, I))` — avoids `inverse_spd` and is numerically stable.
- Data in CSR row format: `row_len[n]`, `row_start[n]`, `obs_col[:]`, `y_vals[:]`. Python helper: `_build_csr_data(Y)` in `befa_stan.py`.

### Pattern-mixture FIML (`befa_missing.stan`)

Groups rows by missingness pattern, one d×d Cholesky per unique pattern per step. Cost: **O(n_patterns · d³)**. Only practical for < ~30% missingness or small P.

### Data augmentation (`befa_augmented.stan`)

Samples N×K factor scores explicitly. Non-centred: `F = F_raw · chol(Ω)'`, `F_raw ~ N(0,I)`. Likelihood: O(n_obs) univariate normals. No Cholesky at all.

**Critical init requirement**: F_raw must be initialised with non-zero values (e.g., `0.5 · N(0,I)`). If F_raw = 0 at init, the likelihood gradient w.r.t. the loadings z is identically 0 (since ∂loglik/∂z ∝ F[n,k]), and all loadings collapse into the horseshoe spike.

---

## 7. Stan model variant selection (`befa_stan.py`)

```python
get_model(variant)          # "dense" | "pattern" | "woodbury" | "augmented"
make_stan_data(..., missing_model="woodbury")   # explicit, never auto-selected
fit_stan(..., missing_model="woodbury")
```

`make_stan_data` selects the Stan variant based on `has_missing` and `missing_model`:
- No NaNs → `"dense"` regardless of `missing_model`
- NaNs + `"woodbury"` → `_build_csr_data(Y)` + Woodbury model
- NaNs + `"pattern"` → `_build_pattern_data(Y)` + pattern-mixture model
- NaNs + `"augmented"` → `_build_sparse_data(Y)` + augmented model

`_make_stan_init` reads `data["_variant"]` and adds `F_raw` only for the augmented model.

`save_warmup=True` is passed to `model.sample()` so the chain CSV files contain warmup rows, enabling the live progress counter in the Streamlit console.

---

## 8. Stan console output

Stan's iteration progress goes to the **real stdout** (file descriptor 1) directly — not through Python's `sys.stdout`. The `_TeeWriter` in `fit_runner.py` wraps `sys.stdout` and captures Python-level `print()` calls from `befa_stan.py`. Stan's own C-binary output bypasses it.

The Streamlit console panel is populated by two sources:
1. **Python prints** (guaranteed captured): compile status, model variant, chain count, completion.
2. **CSV polling thread** (`_poll_stan_csvs`): counts non-comment, non-header rows in new chain CSV files every 2 s and appends/replaces a `[Stan] N/total draws (pct%) — phase` line. Warmup draws appear because `save_warmup=True`.

Phase detection: `phase = "warmup" if n < tune * chains else "sampling"`.

---

## 9. Credible intervals

Computed in `app.py` from the stacked posterior draws:
```python
L_lo = np.percentile(L_draws, ci_lo_pct, axis=0)   # (P, K)
L_hi = np.percentile(L_draws, ci_hi_pct, axis=0)   # (P, K)
```
The same permutation + sign flip applied to `L_aligned` for the truth comparison is also applied to the CI bounds via `_apply_perm_signs`. Sign flips swap lo and hi (correctly handled by `np.minimum / np.maximum`).

`excl 0` flag: `✓` when `lo > 0 or hi < 0` — Bayesian analogue of significance.

CI level: 89 / 90 / 95%, chosen via a selectbox above the tabs.

For **Promax**: `Lambda_post` is replaced by the aligned draws before CIs are computed, so the CIs reflect the spread of aligned draws around the reference — not fully Bayesian uncertainty on Λ. A caption in the Loadings tab explains this.

---

## 10. Code walkthrough, file by file

### `app.py`
Linear Streamlit front-end. Key sections:
- **Sidebar block 1** — data source (simulate or upload CSV)
- **Sidebar block 2** — K, η, slab, identification strategy, anchor items
- **Sidebar block 3** — backend radio, backend-specific hyperparams, missing-data model selector (shown when data has NaNs and CmdStan is selected)
- **Pending-K mechanism** — `pending_fit_K` + `auto_fit` flags drive one-click rerun with suggested K
- **Fit invocation** — `start_fit(prog, backend, **kwargs)` in `fit_runner.py`
- **Result unpacking** — handles both Stan (`stan_fit`) and PyMC (`idata`) result dicts
- **Promax alignment** — called if `identification == "unconstrained"`; replaces `Lambda_post` and `Omega_post` in-place
- **Dead-column detector** — flags columns with SSL < 0.1
- **CI selector** — selectbox + `_apply_perm_signs` helper before the tab block
- **Seven tabs** — Loadings Λ, Factor corr Ω, Uniquenesses ψ, Sampling diag, Recovery, K diagnostic, Save

### `bayesian_efa.py`
PyMC model construction.
- `_resolve_identification(P, K, identification, anchors)` → `(diag_idx, free_idx)`
- `build_model` — standardises Y, builds patterns, declares horseshoe + LKJ + HalfNormal priors, assembles Λ via `pt.set_subtensor`, constructs `Σ`, declares `MvNormal` likelihood (one per pattern if missing)
- `fit_fast` / `fit_full` — thin wrappers for ADVI and NUTS

### `befa.stan` / `befa_missing.stan` / `befa_woodbury.stan` / `befa_augmented.stan`
Four Stan models sharing the same loading structure (horseshoe + LKJ + identification), differing only in the likelihood:
- `befa.stan` — single `multi_normal_cholesky(0, L_Sigma)` over array of vectors
- `befa_missing.stan` — loop over patterns, one sub-block `multi_normal_cholesky` per pattern
- `befa_woodbury.stan` — loop over rows, Woodbury K×K solve per row
- `befa_augmented.stan` — loop over observed cells, one `normal` per cell; `F_raw` parameter

### `befa_stan.py`
cmdstanpy wrapper. Key functions:
- `get_model(variant)` — compiles and caches `CmdStanModel` instances
- `_build_csr_data(Y)` — CSR row format for Woodbury model
- `_build_sparse_data(Y)` — flat (row, col, val) for augmented model
- `_build_pattern_data(Y)` — ragged arrays for pattern-mixture model
- `make_stan_data(Y, K, *, missing_model, ...)` — assembles full data dict, sets `_variant` key
- `_make_stan_init(data, chain_id)` — deterministic near-zero init; adds `F_raw` for augmented
- `fit_stan(...)` — calls `make_stan_data`, `get_model`, `model.sample(save_warmup=True)`

### `fit_runner.py`
Threaded fit runner.
- `FitProgress` dataclass — `status`, `phase`, `iter`, `total`, `log_lines`, `cancel_event`, `thread`
- `_TeeWriter` — wraps `sys.stdout`, buffers partial lines, appends complete lines to `log_lines`
- `_poll_stan_csvs(log_dir, existing_csvs, total_draws, warmup_draws, log_lines, stop_event)` — background thread; counts CSV rows every 2 s; replaces last `[Stan]` line in-place
- `_run_cmdstan` — sets up TeeWriter + poll thread, calls `fit_stan`, tears down on completion
- `start_fit(prog, backend, **kwargs)` — spawns daemon thread

### `simulator.py`
- `make_simple_structure` / `make_factor_corr` / `simulate`

### `diagnostics.py`
- `parallel_analysis` — Horn 1965, NaN-safe
- `loading_strength_diagnostic` — point-estimate SSL
- `shrinkage_factor_count` — per-draw SSL > threshold; returns P(K=k) and per-column P(active)

### `post_process.py`
- `varimax` / `promax` (single-matrix, preserves ΛΦΛᵀ)
- `_reorder_and_flip` / `_chol_psd`
- `promax_align_posterior` — shared-reference alignment (see §5)

---

## 11. Known pathologies and how to talk about them

1. **Empty factor (lower_triangular + horseshoe + ADVI)** — Λ[k,k] ≈ 0.05, column k otherwise zero. Not a bug in loading assembly. Fix: use `unconstrained + Promax` and/or NUTS.

2. **Phantom cross-loadings after per-draw rotation** — per-draw Promax picks slightly different bases → averaging smears. Fix: use the shared-reference Promax (current code).

3. **ADVI LKJ NaN** — `MatrixIsPositiveDefinite` from LKJ during ADVI. Fix: switch to NUTS, increase `lkj_eta`, or reduce K.

4. **Woodbury `cholesky_decompose: A is not symmetric`** — floating-point asymmetry in `M_n = Ω⁻¹ + Λ_S' D_S⁻¹ Λ_S` at extreme proposals. Fixed by symmetrising: `M_n = 0.5 * (M_raw + M_raw')`.

5. **Augmented model loadings collapse to 0** — F_raw initialised to 0 kills the likelihood gradient on loadings. Fixed by initialising F_raw ~ 0.5·N(0,I).

6. **Stan console shows 0 draws** — `save_warmup=False` (old default) means CSV files have no rows during warmup. Fixed by `save_warmup=True`.

---

## 12. Identification-strategy cheat sheet

| Strategy | Bayesian on Λ? | When to use | Known issue |
|---|---|---|---|
| lower_triangular | ✅ Fully | Fast sanity check, well-ordered items | Empty-factor trap (GZ + horseshoe + ADVI) |
| anchor | ✅ Fully | Known K natural anchor items | Bad anchors → same trap |
| unconstrained + Promax | ⚠️ Point-est on Λ; full posterior on Σ_F | Exploratory, want clean simple structure | CIs on Λ are not fully Bayesian |

A fourth strategy — **CUSP** (Legramanti, Durante, Dunson 2020) — uses cumulative shrinkage to dynamically kill higher-index columns. This is the right fully-Bayesian sparsity prior without post-hoc rotation. Not yet implemented.

---

## 13. Sanity tests

```bash
python test_recovery.py    # PyMC fit smoke test
python test_stan.py        # CmdStan smoke test
```

**Promax algebra invariant**:
```python
from post_process import promax
import numpy as np
Lu = np.array([[0.8,0,0],[0.8,0,0],[0,0.8,0],[0,0.8,0],[0,0,0.8],[0,0,0.8]], float)
Ls, Phi = promax(Lu)
assert np.allclose(Ls @ Phi @ Ls.T, Lu @ Lu.T), "promax broke Λ Φ Λᵀ = Lu Luᵀ"
```

---

## 14. Key variable names

| Name | Shape | Meaning |
|---|---|---|
| `Y` | (N, P) | Data matrix (standardised, possibly with NaNs) |
| `Lambda` / `Λ` | (P, K) | Loading matrix |
| `Omega` / `Ω` | (K, K) | Factor correlation matrix |
| `psi` / `ψ` | (P,) | Uniqueness standard deviations |
| `Lambda_post` | (chains, draws, P, K) | Posterior draws of Λ |
| `Omega_post` | (chains, draws, K, K) | Posterior draws of Ω |
| `diag_idx` | list[(row, col)] | Positivity-constrained loadings |
| `free_idx` | list[(row, col)] | Shrinkage-prior loadings |
| `tau0` | scalar | Horseshoe global-shrinkage scale heuristic |
| `slab_scale`, `slab_df` | scalar | Regularised-horseshoe slab parameters |
| `lkj_eta` | scalar | LKJ concentration on Ω |
| `truth` | dict | `{Lambda, Omega, psi, F}` — simulator ground truth |
| `perm` | list[int] | Procrustes column permutation (truth alignment) |
| `signs` | (K,) | Procrustes sign flips (truth alignment) |

---

## 15. What to do when the user asks for a new feature

1. **Model change** → edit `build_model` in `bayesian_efa.py` AND the relevant `.stan` file(s). Keep PyMC and Stan mathematically identical.
2. **New identification strategy** → edit `_resolve_identification` + `_build_indices` in `befa_stan.py`. Keep `lower_triangular` / `anchor` / `unconstrained` as the base strategies.
3. **New Stan likelihood variant** → add `.stan` file, add `"variant_name"` to `get_model`, add `_build_*_data` helper, add branch in `make_stan_data`.
4. **Post-hoc analysis** → add to `post_process.py` or `diagnostics.py`, wire into a tab in `app.py`.
5. **New backend** → add worker in `fit_runner.py`, radio option in `app.py`, result-unpack branch after fit completion.
6. **Always** run Promax sanity test (§13) if touching `post_process.py`.
7. **Never** remove identification strategies — the trichotomy is a core feature.
8. **Streamlit widget rule** — never write to `st.session_state[key]` when `key` is also a widget key that has already rendered. Use `pending_<name>` and consume at the top of the next run.

---

*Last meaningful update: added Woodbury FIML model (befa_woodbury.stan), made missing-data model selection explicit in the UI, added credible intervals with `excl 0` flag to all posterior tables, fixed Stan console polling to use CSV row counting with `save_warmup=True`, fixed data-augmentation F_raw init collapse.*
