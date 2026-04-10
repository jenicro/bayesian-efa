"""Standalone overnight fit runner — no Streamlit required.

The script saves Y (the data), ground truth (if simulated), and the full
posterior to an output directory before and after sampling.  If the process
is killed mid-run the Stan CSV files are still on disk and can be recovered
with recover_stan_fit.py.

Usage — simulate:
    python run_overnight.py simulate \\
        --K 3 --N 500 --items-per-factor 5 5 5 \\
        --chains 4 --warmup 1000 --draws 2000 \\
        --outdir results/sim_run1

Usage — CSV:
    python run_overnight.py csv data/my_data.csv \\
        --K 4 --chains 4 --warmup 1000 --draws 2000 \\
        --outdir results/csv_run1

All posteriors are saved to --outdir as:
  Y.csv              — data matrix (with column names)
  truth.npz          — ground truth Lambda/Omega/psi/F  (simulate only)
  sim_params.json    — simulation parameters             (simulate only)
  Lambda_post.npy    — (chains, draws, P, K)
  Omega_post.npy     — (chains, draws, K, K)
  psi_post.npy       — (chains, draws, P)
  all_draws.csv      — long-format draws
  summary.csv        — R-hat / ESS per parameter
  meta.json          — fit hyperparameters + column names
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── simulate ──────────────────────────────────────────────────────────
    sim = sub.add_parser("simulate", help="Simulate data and fit.")
    sim.add_argument("--K", type=int, required=True, help="Number of factors.")
    sim.add_argument("--N", type=int, default=500, help="Number of observations.")
    sim.add_argument(
        "--items-per-factor", type=int, nargs="+", default=None,
        metavar="N",
        help="Items per factor (space-separated). Length must equal K. "
             "Default: 5 items per factor.",
    )
    sim.add_argument("--main-loading-lo", type=float, default=0.6)
    sim.add_argument("--main-loading-hi", type=float, default=0.85)
    sim.add_argument("--cross-loading-prob", type=float, default=0.0)
    sim.add_argument("--rho", type=float, default=0.3,
                     help="Equicorrelation between factors.")
    sim.add_argument("--psi-lo", type=float, default=0.2)
    sim.add_argument("--psi-hi", type=float, default=0.5)
    sim.add_argument("--sim-seed", type=int, default=0)

    # ── csv ───────────────────────────────────────────────────────────────
    csv_p = sub.add_parser("csv", help="Load data from a CSV file and fit.")
    csv_p.add_argument("csv_path", help="Path to CSV file (rows=obs, cols=items).")
    csv_p.add_argument("--K", type=int, required=True, help="Number of factors.")
    csv_p.add_argument(
        "--no-header", action="store_true",
        help="CSV has no header row; columns named x1..xP.",
    )

    # ── shared fit args ───────────────────────────────────────────────────
    for s in (sim, csv_p):
        s.add_argument("--outdir", default="overnight_results",
                       help="Output directory.")
        s.add_argument("--chains", type=int, default=4)
        s.add_argument("--warmup", type=int, default=1000)
        s.add_argument("--draws", type=int, default=1000,
                       help="Sampling draws PER CHAIN.")
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--adapt-delta", type=float, default=0.95)
        s.add_argument("--max-treedepth", type=int, default=10)
        s.add_argument(
            "--identification",
            choices=["lower_triangular", "anchor", "unconstrained"],
            default="lower_triangular",
        )
        s.add_argument("--lkj-eta", type=float, default=2.0)
        s.add_argument("--slab-scale", type=float, default=2.5)
        s.add_argument("--slab-df", type=float, default=4.0)
        s.add_argument(
            "--missing-model",
            choices=["woodbury", "pattern", "augmented"],
            default="woodbury",
        )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_long_csv(Lambda_post, Omega_post, psi_post, col_names, out: Path):
    """Long-format CSV: param, draw, row, col, value."""
    chains, draws, P, K = Lambda_post.shape
    D = chains * draws
    Lf  = Lambda_post.reshape(D, P, K)
    Of  = Omega_post.reshape(D, K, K)
    Pf  = psi_post.reshape(D, P)

    f_labels = [f"F{k+1}" for k in range(K)]
    rows = []

    draw_idx = np.repeat(np.arange(D), P * K)
    r_L = np.tile(np.repeat(col_names, K), D)
    c_L = np.tile(f_labels, D * P)
    v_L = Lf.reshape(-1)
    lambda_df = pd.DataFrame({"param": "Lambda", "draw": draw_idx,
                               "row": r_L, "col": c_L, "value": v_L})

    draw_idx_o = np.repeat(np.arange(D), K * K)
    r_O = np.tile(np.repeat(f_labels, K), D)
    c_O = np.tile(f_labels, D * K)
    v_O = Of.reshape(-1)
    omega_df = pd.DataFrame({"param": "Omega", "draw": draw_idx_o,
                              "row": r_O, "col": c_O, "value": v_O})

    draw_idx_p = np.repeat(np.arange(D), P)
    r_P = np.tile(col_names, D)
    v_P = Pf.reshape(-1)
    psi_df = pd.DataFrame({"param": "psi", "draw": draw_idx_p,
                            "row": r_P, "col": "", "value": v_P})

    df = pd.concat([lambda_df, omega_df, psi_df], ignore_index=True)
    path = out / "all_draws.csv"
    df.to_csv(path, index=False)
    print(f"[run] wrote {path} ({len(df):,} rows)")


def extract_posteriors(fit, chains: int):
    """Extract (chains, draws, ...) arrays from a cmdstanpy fit."""
    Lambda = fit.stan_variable("Lambda")   # (total_draws, P, K)
    Omega  = fit.stan_variable("Omega")    # (total_draws, K, K)
    psi    = fit.stan_variable("psi")      # (total_draws, P)

    total_draws = Lambda.shape[0]
    draws_per   = total_draws // chains

    Lambda_post = Lambda.reshape(chains, draws_per, *Lambda.shape[1:])
    Omega_post  = Omega.reshape(chains, draws_per, *Omega.shape[1:])
    psi_post    = psi.reshape(chains, draws_per, *psi.shape[1:])
    return Lambda_post, Omega_post, psi_post


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────
    truth = None
    if args.mode == "simulate":
        from simulator import make_simple_structure, make_factor_corr, simulate

        K = args.K
        ipf = args.items_per_factor or [5] * K
        if len(ipf) != K:
            sys.exit(f"[error] --items-per-factor has {len(ipf)} values but K={K}")

        Lambda_true = make_simple_structure(
            ipf,
            main_loading_range=(args.main_loading_lo, args.main_loading_hi),
            cross_loading_prob=args.cross_loading_prob,
            rng=np.random.default_rng(args.sim_seed),
        )
        Omega_true = make_factor_corr(K, args.rho)
        Y, truth = simulate(
            Lambda_true, Omega_true, args.N,
            psi_range=(args.psi_lo, args.psi_hi),
            seed=args.sim_seed,
        )
        P = Y.shape[1]
        col_names = [f"x{j+1}" for j in range(P)]

        # Save ground truth before fit starts
        np.savez(out / "truth.npz", **{k: v for k, v in truth.items()})
        sim_params = dict(
            K=K, N=args.N, items_per_factor=ipf,
            main_loading_range=[args.main_loading_lo, args.main_loading_hi],
            cross_loading_prob=args.cross_loading_prob,
            rho=args.rho,
            psi_range=[args.psi_lo, args.psi_hi],
            sim_seed=args.sim_seed,
        )
        (out / "sim_params.json").write_text(json.dumps(sim_params, indent=2))
        print(f"[run] simulated N={args.N}, P={P}, K={K}")
        print(f"[run] truth saved → {out / 'truth.npz'}")

    else:  # csv
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            sys.exit(f"[error] CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, header=None if args.no_header else 0)
        col_names = [f"x{j+1}" for j in range(df.shape[1])] if args.no_header \
                    else list(df.columns.astype(str))
        Y = df.values.astype(float)
        K = args.K
        print(f"[run] loaded CSV: {csv_path.name}  shape={Y.shape}")

    # Save data matrix (always, before fit)
    pd.DataFrame(Y, columns=col_names).to_csv(out / "Y.csv", index=False)
    print(f"[run] data saved → {out / 'Y.csv'}")

    # ── 2. Fit metadata ───────────────────────────────────────────────────
    meta = dict(
        mode=args.mode,
        K=K,
        col_names=col_names,
        chains=args.chains,
        warmup=args.warmup,
        draws=args.draws,
        seed=args.seed,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        identification=args.identification,
        lkj_eta=args.lkj_eta,
        slab_scale=args.slab_scale,
        slab_df=args.slab_df,
        missing_model=args.missing_model,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[run] meta saved → {out / 'meta.json'}")

    # ── 3. Fit ────────────────────────────────────────────────────────────
    from befa_stan import fit_stan

    print(f"\n[run] starting CmdStan fit  "
          f"({args.chains} chains × {args.warmup}+{args.draws} steps) …\n")

    fit = fit_stan(
        Y, K,
        chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.draws,
        seed=args.seed,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        identification=args.identification,
        lkj_eta=args.lkj_eta,
        slab_scale=args.slab_scale,
        slab_df=args.slab_df,
        missing_model=args.missing_model,
        show_progress=False,
        show_console=True,
    )

    # ── 4. Extract & save posteriors ──────────────────────────────────────
    print("\n[run] extracting posteriors …")
    Lambda_post, Omega_post, psi_post = extract_posteriors(fit, args.chains)

    np.save(out / "Lambda_post.npy", Lambda_post)
    np.save(out / "Omega_post.npy",  Omega_post)
    np.save(out / "psi_post.npy",    psi_post)
    print(f"[run] posteriors saved  "
          f"Lambda={Lambda_post.shape}  Omega={Omega_post.shape}  psi={psi_post.shape}")

    save_long_csv(Lambda_post, Omega_post, psi_post, col_names, out)

    # ── 5. Sampling summary ───────────────────────────────────────────────
    s = fit.summary()
    s.to_csv(out / "summary.csv")
    sub = s.loc[s.index.str.startswith(("Lambda", "Omega", "psi"))]
    if not sub.empty and "R_hat" in sub.columns:
        print(f"\n=== Sampling diagnostics ===")
        print(f"  max R̂        = {sub['R_hat'].max():.3f}")
        if "ESS_bulk" in sub.columns:
            print(f"  min ESS_bulk = {sub['ESS_bulk'].min():.0f}")
        if "ESS_tail" in sub.columns:
            print(f"  min ESS_tail = {sub['ESS_tail'].min():.0f}")
    print(f"[run] summary saved → {out / 'summary.csv'}")

    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] all outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
