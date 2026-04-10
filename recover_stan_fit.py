"""Recover a Streamlit-style report from an orphaned CmdStan output directory.

Usage
-----
    python recover_stan_fit.py <cmdstan_output_dir> [--identification lower_triangular|unconstrained]
                                                    [--var-names "x1,x2,..."]
                                                    [--outdir recovered_fit]

When the Streamlit session that started a CmdStan fit dies (browser closed,
etc.) the Stan subprocesses keep going and write their draws to a temp
directory — typically ``%TEMP%\\tmp<random>``. This script loads those CSVs
via ``cmdstanpy.from_csv`` and reproduces everything from the Streamlit
app that doesn't require the original data matrix Y:

  • Loading matrix Λ  (heatmap + CSV table + sd)
  • Factor correlation Ω  (heatmap)
  • Uniquenesses ψ  (bar)
  • R̂ / ESS summary
  • Dead-factor detector (posterior-mean SSL < 0.1)
  • Shrinkage-based factor-count diagnostic
  • Optional Promax shared-reference post-processing
    (when ``--identification unconstrained``)
  • Long-CSV dump of all draws

What is NOT recoverable
-----------------------
Without the original Y:
  • empirical correlation matrix
  • parallel analysis / scree plot
  • recovery-vs-truth (needs simulated truth)
  • standardization scale (so Λ is on whatever scale the fit used)

All outputs are written to ``--outdir`` (default: ./recovered_fit).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import from_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stan_dir",
                   help="Path to the cmdstanpy output directory "
                        "(contains befa-*_<chain>.csv files).")
    p.add_argument("--identification",
                   choices=["lower_triangular", "anchor", "unconstrained"],
                   default="lower_triangular",
                   help="Identification used during sampling. If 'unconstrained', "
                        "the shared-reference Promax post-processing is applied.")
    p.add_argument("--var-names", default=None,
                   help="Comma-separated variable labels for Λ rows "
                        "(defaults to x1..xP).")
    p.add_argument("--outdir", default="recovered_fit",
                   help="Where to write plots and CSVs.")
    p.add_argument("--kappa", type=float, default=4.0,
                   help="Promax kappa (only used if identification=unconstrained).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def load_fit(stan_dir: str):
    """Load a cmdstanpy fit from a directory of CSVs."""
    d = Path(stan_dir)
    if not d.exists():
        sys.exit(f"[error] directory does not exist: {d}")
    # Recursively find chain CSVs: files named like
    # "<model>-<timestamp>_<chain>.csv". Skip stansummary/diagnose.
    import re
    pattern = re.compile(r"_\d+\.csv$")
    all_csvs = [p for p in d.rglob("*.csv")
                if pattern.search(p.name)
                and not p.name.startswith(("stansummary", "diagnose"))]
    if not all_csvs:
        sys.exit(f"[error] no chain CSV files found under {d}")

    # Group by parent dir + timestamp prefix so multiple fits don't mix.
    def fit_key(p: Path):
        m = re.match(r"(.*)_\d+\.csv$", p.name)
        return (str(p.parent), m.group(1) if m else p.name)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for p in all_csvs:
        groups[fit_key(p)].append(p)

    # Pick the group with the most recent mtime.
    best_key = max(groups,
                   key=lambda k: max(p.stat().st_mtime for p in groups[k]))
    chain_csvs = sorted(groups[best_key])
    if len(groups) > 1:
        print(f"[info] found {len(groups)} fit group(s); "
              f"using the most recent one ({best_key[1]})")
    print(f"[info] loading {len(chain_csvs)} chain CSVs")
    for p in chain_csvs:
        print(f"       {p}")
    fit = from_csv([str(p) for p in chain_csvs])
    return fit


def extract_posteriors(fit):
    """Return Lambda_post, Omega_post, psi_post with shape prefix (chains=1, draws, ...)."""
    Lambda = fit.stan_variable("Lambda")
    Omega  = fit.stan_variable("Omega")
    psi    = fit.stan_variable("psi")
    # cmdstanpy returns (draws, ...) with chains stacked. Add a fake chain dim
    # so the code below matches the app's expected (chains, draws, P, K) shape.
    Lambda_post = Lambda[None, ...]
    Omega_post  = Omega[None, ...]
    psi_post    = psi[None, ...]
    print(f"[info] Lambda shape: {Lambda_post.shape}  "
          f"Omega: {Omega_post.shape}  psi: {psi_post.shape}")
    return Lambda_post, Omega_post, psi_post


def plot_lambda(L_mean: np.ndarray, var_names: list[str], out: Path):
    P, K = L_mean.shape
    fig, ax = plt.subplots(figsize=(1.0 * K + 2, 0.32 * P + 1))
    im = ax.imshow(L_mean, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(K), [f"F{k+1}" for k in range(K)])
    ax.set_yticks(range(P), var_names)
    for j in range(P):
        for k in range(K):
            ax.text(k, j, f"{L_mean[j, k]:.2f}", ha="center", va="center",
                    color="white" if abs(L_mean[j, k]) > 0.4 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Loadings Λ (posterior mean)")
    fig.tight_layout()
    fig.savefig(out / "loadings.png", dpi=150)
    plt.close(fig)
    print(f"[info] wrote {out / 'loadings.png'}")


def plot_omega(O_mean: np.ndarray, out: Path):
    K = O_mean.shape[0]
    fig, ax = plt.subplots(figsize=(3 + 0.3 * K, 3 + 0.3 * K))
    im = ax.imshow(O_mean, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{O_mean[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(K), [f"F{k+1}" for k in range(K)])
    ax.set_yticks(range(K), [f"F{k+1}" for k in range(K)])
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Factor correlation Ω (posterior mean)")
    fig.tight_layout()
    fig.savefig(out / "omega.png", dpi=150)
    plt.close(fig)
    print(f"[info] wrote {out / 'omega.png'}")


def plot_psi(psi_mean: np.ndarray, psi_sd: np.ndarray,
             var_names: list[str], out: Path):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(psi_mean)), psi_mean, yerr=psi_sd, color="steelblue")
    ax.set_xticks(range(len(psi_mean)), var_names, rotation=45, ha="right")
    ax.set_ylabel("ψ (residual sd)")
    ax.set_title("Uniquenesses ψ (posterior mean ± sd)")
    fig.tight_layout()
    fig.savefig(out / "psi.png", dpi=150)
    plt.close(fig)
    print(f"[info] wrote {out / 'psi.png'}")


def dead_column_check(L_mean: np.ndarray):
    ssl = (L_mean ** 2).sum(axis=0)
    dead = [k for k, s in enumerate(ssl) if s < 0.1]
    print("\n=== Column SSL (sum of squared loadings) ===")
    for k, s in enumerate(ssl):
        tag = " ← DEAD" if k in dead else ""
        print(f"  F{k+1}: {s:.4f}{tag}")
    if dead:
        print(f"[warn] {len(dead)} dead column(s); effective K ≈ {len(ssl) - len(dead)}")


def shrinkage_diagnostic(Lambda_post: np.ndarray):
    ssl_draws = (Lambda_post ** 2).sum(axis=-2)
    active = ssl_draws > 1.0
    prob_active = active.mean(axis=(0, 1))
    counts = active.sum(axis=-1).reshape(-1)
    print("\n=== Shrinkage-based factor count ===")
    print(f"  posterior mean K  = {counts.mean():.2f}")
    print(f"  posterior median K = {int(np.median(counts))}")
    print(f"  per-column P[SSL>1]:")
    for k, p in enumerate(prob_active):
        print(f"    F{k+1}: {p:.3f}")
    print(f"  suggested K = {int((prob_active > 0.5).sum())}")


def save_long_csv(Lambda_post, Omega_post, psi_post,
                  var_names: list[str], out: Path):
    _, D, P, K = Lambda_post.shape
    rows = []
    Lf = Lambda_post.reshape(D, P, K)
    Of = Omega_post.reshape(D, K, K)
    Pf = psi_post.reshape(D, P)
    for d in range(D):
        for j in range(P):
            for k in range(K):
                rows.append(("Lambda", d, var_names[j], f"F{k+1}",
                             float(Lf[d, j, k])))
        for i in range(K):
            for j in range(K):
                rows.append(("Omega", d, f"F{i+1}", f"F{j+1}",
                             float(Of[d, i, j])))
        for j in range(P):
            rows.append(("psi", d, var_names[j], "", float(Pf[d, j])))
    df = pd.DataFrame(rows, columns=["param", "draw", "row", "col", "value"])
    path = out / "all_draws.csv"
    df.to_csv(path, index=False)
    print(f"[info] wrote {path} ({len(df):,} rows)")


def save_summaries(fit, out: Path):
    s = fit.summary()
    s.to_csv(out / "summary.csv")
    print(f"[info] wrote {out / 'summary.csv'}")
    sub = s.loc[s.index.str.startswith(("Lambda", "Omega", "psi"))]
    if not sub.empty and "R_hat" in sub.columns:
        print(f"\n=== Sampling diagnostics ===")
        print(f"  max R̂        = {sub['R_hat'].max():.3f}")
        if "ESS_bulk" in sub.columns:
            print(f"  min ESS_bulk = {sub['ESS_bulk'].min():.0f}")
        if "ESS_tail" in sub.columns:
            print(f"  min ESS_tail = {sub['ESS_tail'].min():.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fit = load_fit(args.stan_dir)
    Lambda_post, Omega_post, psi_post = extract_posteriors(fit)

    _, D, P, K = Lambda_post.shape
    var_names = (args.var_names.split(",") if args.var_names
                 else [f"x{j+1}" for j in range(P)])
    if len(var_names) != P:
        sys.exit(f"[error] --var-names has {len(var_names)} entries but P={P}")

    # Optional Promax alignment for unconstrained fits
    if args.identification == "unconstrained":
        print("[info] applying shared-reference Promax alignment "
              f"(kappa={args.kappa})")
        from post_process import promax_align_posterior
        Lambda_post, Omega_post = promax_align_posterior(
            Lambda_post, Omega_post, kappa=args.kappa
        )

    # Posterior means / sds
    L_mean = Lambda_post.mean(axis=(0, 1))
    L_sd   = Lambda_post.std(axis=(0, 1))
    O_mean = Omega_post.mean(axis=(0, 1))
    psi_mean = psi_post.mean(axis=(0, 1))
    psi_sd   = psi_post.std(axis=(0, 1))

    # Plots
    plot_lambda(L_mean, var_names, out)
    plot_omega(O_mean, out)
    plot_psi(psi_mean, psi_sd, var_names, out)

    # Tables
    L_table = pd.DataFrame({
        "var":    np.repeat(var_names, K),
        "factor": np.tile([f"F{k+1}" for k in range(K)], P),
        "mean":   L_mean.flatten(),
        "sd":     L_sd.flatten(),
    })
    L_table.to_csv(out / "loadings.csv", index=False)
    pd.DataFrame(O_mean,
                 index=[f"F{k+1}" for k in range(K)],
                 columns=[f"F{k+1}" for k in range(K)]).to_csv(out / "omega.csv")
    pd.DataFrame({"var": var_names, "psi_mean": psi_mean, "psi_sd": psi_sd}
                 ).to_csv(out / "psi.csv", index=False)
    print(f"[info] wrote loadings.csv, omega.csv, psi.csv")

    # Diagnostics
    dead_column_check(L_mean)
    shrinkage_diagnostic(Lambda_post)
    save_summaries(fit, out)
    save_long_csv(Lambda_post, Omega_post, psi_post, var_names, out)

    print(f"\n[done] everything written to {out.resolve()}")


if __name__ == "__main__":
    main()
