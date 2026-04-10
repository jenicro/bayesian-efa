"""Streamlit viewer for a recovered CmdStan fit.

Run:
    python -m streamlit run view_recovered.py -- --dir recovered_fit

Loads the CSVs written by recover_stan_fit.py and shows Loadings, Ω, ψ,
diagnostics, and the raw summary in the same tabbed layout as app.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def _get_dir() -> Path:
    # Parse after `--`
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="recovered_fit")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args, _ = ap.parse_known_args(argv)
    return Path(args.dir)


st.set_page_config(page_title="Recovered BEFA Fit", layout="wide")
st.title("🛟 Recovered BEFA Fit Viewer")

d = _get_dir()
st.caption(f"Source directory: `{d.resolve()}`")

if not d.exists():
    st.error(f"Directory not found: {d}")
    st.stop()

# ---- Load tables ----------------------------------------------------
try:
    loadings_df = pd.read_csv(d / "loadings.csv")
    omega_df    = pd.read_csv(d / "omega.csv", index_col=0)
    psi_df      = pd.read_csv(d / "psi.csv")
    summary_df  = pd.read_csv(d / "summary.csv", index_col=0)
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Did you run `recover_stan_fit.py` first?")
    st.stop()

all_draws_path = d / "all_draws.csv"
have_draws = all_draws_path.exists()

# Reconstruct matrices
factors = sorted(loadings_df["factor"].unique(),
                 key=lambda x: int(x.replace("F", "")))
vars_   = list(dict.fromkeys(loadings_df["var"]))
P, K = len(vars_), len(factors)
L_mean = loadings_df.pivot(index="var", columns="factor",
                           values="mean").reindex(vars_)[factors].values
L_sd   = loadings_df.pivot(index="var", columns="factor",
                           values="sd").reindex(vars_)[factors].values
O_mean = omega_df.values
psi_mean = psi_df["psi_mean"].values
psi_sd   = psi_df["psi_sd"].values

st.success(f"Loaded posterior summaries · P = {P}, K = {K}")

# ---- Top KPIs -------------------------------------------------------
col_ssl = (L_mean ** 2).sum(axis=0)
dead = [k for k, s in enumerate(col_ssl) if s < 0.1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Variables (P)", P)
c2.metric("Factors (K)", K)
c3.metric("Dead columns", len(dead),
          delta=f"SSL<0.1" if dead else "none",
          delta_color="inverse" if dead else "normal")
if "R_hat" in summary_df.columns:
    try:
        max_rhat = summary_df["R_hat"].dropna().max()
        c4.metric("max R̂", f"{float(max_rhat):.3f}")
    except Exception:
        pass

if dead:
    st.warning(
        f"⚠️ {len(dead)} column(s) with SSL < 0.1: "
        + ", ".join(f"F{k+1} (SSL={col_ssl[k]:.3f})" for k in dead)
        + ". Consider refitting with smaller K."
    )

# ---- Tabs -----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Loadings Λ", "Factor corr Ω", "Uniquenesses ψ",
     "Sampling diag", "K diagnostic", "Raw tables"]
)

with tab1:
    st.subheader("Posterior-mean loadings Λ")
    fig, ax = plt.subplots(figsize=(1.0 * K + 2, 0.32 * P + 1))
    im = ax.imshow(L_mean, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(K), factors)
    ax.set_yticks(range(P), vars_)
    for j in range(P):
        for k in range(K):
            ax.text(k, j, f"{L_mean[j, k]:.2f}", ha="center", va="center",
                    color="white" if abs(L_mean[j, k]) > 0.4 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)
    st.dataframe(loadings_df, hide_index=True)

with tab2:
    st.subheader("Factor correlation Ω")
    fig, ax = plt.subplots(figsize=(3 + 0.3 * K, 3 + 0.3 * K))
    im = ax.imshow(O_mean, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{O_mean[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(K), factors)
    ax.set_yticks(range(K), factors)
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)
    st.dataframe(omega_df)

with tab3:
    st.subheader("Uniquenesses ψ (residual sd)")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(P), psi_mean, yerr=psi_sd, color="steelblue")
    ax.set_xticks(range(P), vars_, rotation=45, ha="right")
    ax.set_ylabel("ψ")
    st.pyplot(fig, clear_figure=True)
    st.dataframe(psi_df, hide_index=True)

with tab4:
    st.subheader("R̂ / ESS")
    sub = summary_df.loc[
        summary_df.index.str.startswith(("Lambda", "Omega", "psi"))
    ]
    cols_show = [c for c in ["Mean", "StdDev", "R_hat", "ESS_bulk", "ESS_tail"]
                 if c in sub.columns]
    st.dataframe(sub[cols_show] if cols_show else sub)
    if "R_hat" in sub.columns:
        rhat = pd.to_numeric(sub["R_hat"], errors="coerce").dropna()
        st.write(f"**max R̂** = {rhat.max():.3f} · "
                 f"**# R̂ > 1.05** = {(rhat > 1.05).sum()}")
    if "ESS_bulk" in sub.columns:
        ess = pd.to_numeric(sub["ESS_bulk"], errors="coerce").dropna()
        st.write(f"**min ESS_bulk** = {ess.min():.0f}")

with tab5:
    st.subheader("Factor count diagnostics")
    st.caption("Computed from the posterior draws (requires all_draws.csv).")
    if not have_draws:
        st.info("No `all_draws.csv` found — the shrinkage diagnostic needs it. "
                "Re-run `recover_stan_fit.py` to produce it.")
    else:
        draws = pd.read_csv(all_draws_path)
        L_long = draws[draws["param"] == "Lambda"]
        D = L_long["draw"].nunique()
        L_draws = (L_long.pivot_table(
            index="draw", columns=["row", "col"], values="value"
        ).values.reshape(D, P, K))
        ssl_draws = (L_draws ** 2).sum(axis=1)             # (D, K)
        active = ssl_draws > 1.0
        counts = active.sum(axis=1)
        prob_active = active.mean(axis=0)
        suggested_K = int((prob_active > 0.5).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Posterior-mean SSL > 1", int((col_ssl > 1.0).sum()))
        c2.metric("Shrinkage P[SSL>1]>0.5", suggested_K)
        c3.metric("Posterior median K", int(np.median(counts)))

        st.markdown("**Per-column activation probability P[SSL > 1]**")
        st.dataframe(pd.DataFrame({
            "factor": factors,
            "P[active]": prob_active,
            "posterior-mean SSL": ssl_draws.mean(axis=0),
        }), hide_index=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        vals, edges = np.histogram(counts, bins=np.arange(K + 2) - 0.5)
        ax.bar(np.arange(K + 1), vals / vals.sum(), color="steelblue")
        ax.set_xlabel("# active factors (SSL > 1)")
        ax.set_ylabel("posterior prob")
        ax.set_title("Posterior distribution of active-factor count")
        st.pyplot(fig, clear_figure=True)

with tab6:
    st.subheader("Full cmdstanpy summary")
    st.dataframe(summary_df)
