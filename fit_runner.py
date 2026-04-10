"""Threaded fit runner with live progress + cancellation.

Used by app.py to keep the Streamlit UI responsive, surface
iteration counts, and let the user abort a long-running fit.
"""
from __future__ import annotations

import collections
import glob
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

_LOG_MAXLEN = 500   # max lines kept in the UI console ring-buffer


@dataclass
class FitProgress:
    status: str = "idle"           # idle | running | done | cancelled | error
    backend: str = ""
    iter: int = 0
    total: int = 0
    phase: str = ""                # "ADVI" | "warmup" | "sampling" | "compiling" | "stan"
    message: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    result: Any = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    log_lines: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_LOG_MAXLEN)
    )

    def request_cancel(self):
        self.cancel_event.set()
        self.message = "cancel requested…"

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at if self.started_at else 0.0


class _Cancelled(Exception):
    pass


class _TeeWriter:
    """Write to both the original stdout and a list (for live UI display).

    Buffers partial lines so only complete lines are appended, keeping the
    list consistent for thread-safe reads from the main Streamlit thread.
    """
    def __init__(self, original, lines: list):
        self._orig = original
        self._lines = lines
        self._partial = ""

    def write(self, text: str) -> int:
        self._orig.write(text)
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._lines.append(line)
        return len(text)

    def flush(self):
        self._orig.flush()

    def fileno(self):            # delegate so subprocess plumbing still works
        return self._orig.fileno()


# ---------- ADVI callback ---------------------------------------------
class _ADVITracker:
    def __init__(self, prog: FitProgress, total: int):
        self.prog = prog
        self.total = total

    def __call__(self, approx, loss, i):
        if self.prog.cancel_event.is_set():
            raise _Cancelled()
        # i is 1-indexed iteration count
        self.prog.iter = int(i)
        self.prog.total = int(self.total)
        if loss is not None and len(loss) > 0:
            self.prog.message = f"ELBO ≈ {-float(loss[-1]):.1f}"


# ---------- NUTS callback ---------------------------------------------
def _make_nuts_callback(prog: FitProgress, total_per_chain: int, chains: int):
    counters = {"draws": 0, "chain": 1}

    def cb(trace, draw):
        if prog.cancel_event.is_set():
            raise _Cancelled()
        counters["draws"] += 1
        prog.iter = counters["draws"]
        prog.total = total_per_chain * chains
        # rough phase tag based on draw.tuning attribute if present
        prog.phase = "warmup" if getattr(draw, "tuning", False) else "sampling"

    return cb


# ---------- Workers ---------------------------------------------------
def _run_pymc_advi(prog, Y, K, n_iter, lkj_eta, slab_scale, seed,
                   identification="lower_triangular", anchors=None,
                   orthogonal=False):
    import pymc as pm
    from bayesian_efa import build_model

    prog.phase = "compiling"
    prog.message = "building model…"
    model = build_model(
        Y, K, lkj_eta=lkj_eta, slab_scale=slab_scale,
        identification=identification, anchors=anchors,
        orthogonal=orthogonal,
    )
    prog.phase = "ADVI"
    prog.message = "fitting…"
    tracker = _ADVITracker(prog, n_iter)
    with model:
        approx = pm.fit(
            n=n_iter,
            method="advi",
            random_seed=seed,
            progressbar=False,
            callbacks=[tracker],
        )
        if prog.cancel_event.is_set():
            raise _Cancelled()
        idata = approx.sample(1000)
    return {"idata": idata, "elbo": approx.hist, "model": model}


def _run_pymc_nuts(prog, Y, K, draws, tune, chains, lkj_eta, slab_scale, seed,
                   identification="lower_triangular", anchors=None,
                   orthogonal=False):
    import pymc as pm
    from bayesian_efa import build_model

    prog.phase = "compiling"
    prog.message = "building model…"
    model = build_model(
        Y, K, lkj_eta=lkj_eta, slab_scale=slab_scale,
        identification=identification, anchors=anchors,
        orthogonal=orthogonal,
    )
    cb = _make_nuts_callback(prog, draws + tune, chains)
    prog.phase = "warmup"
    prog.message = "sampling…"
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,                # single-process so the callback runs in-thread
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
            callback=cb,
            idata_kwargs={"log_likelihood": False},
        )
    return {"idata": idata, "model": model}


def _poll_stan_csvs(log_dir, existing_csvs, total_draws, warmup_draws, log_lines, stop_event):
    """Background thread: track iteration progress by tailing chain CSV files.

    Stan writes one CSV row per draw (warmup + sampling, because save_warmup=True).
    We read only the *new bytes* since the last poll (incremental seek) so the
    read cost is O(new_rows) not O(total_file_size) — critical for large P/K runs
    where each row can be several KB.

    The latest progress line is replaced in-place so the deque stays readable.
    Python's GIL makes single deque[-1] assignment thread-safe.
    """
    log_dir     = Path(log_dir)
    csv_files:  list[str] = []
    file_counts: dict[str, int] = {}   # cumulative data-row count per file
    file_offsets: dict[str, int] = {}  # byte offset of last read per file

    def _is_data_line(raw: str) -> bool:
        s = raw.lstrip()
        return bool(s) and not s.startswith("#") and not s.startswith("lp__")

    while not stop_event.wait(timeout=2.0):
        # Discover new chain CSV files for this run
        if not csv_files:
            all_now = set(glob.glob(str(log_dir / "*.csv")))
            csv_files = sorted(all_now - existing_csvs)
            if csv_files:
                log_lines.append(
                    f"[Stan] {len(csv_files)} chain file(s) opened — tracking draws…"
                )

        if not csv_files:
            continue

        # Read only the bytes written since the last poll
        for fp in csv_files:
            offset = file_offsets.get(fp, 0)
            try:
                with open(fp, "r", errors="replace") as f:
                    f.seek(offset)
                    new_text = f.read()
                    file_offsets[fp] = f.tell()
                new_rows = sum(1 for line in new_text.splitlines() if _is_data_line(line))
                file_counts[fp] = file_counts.get(fp, 0) + new_rows
            except OSError:
                pass

        n = sum(file_counts.values())
        pct   = min(100, int(100 * n / max(1, total_draws)))
        phase = "warmup" if n < warmup_draws else "sampling"
        msg   = f"[Stan] {n}/{total_draws} draws ({pct}%) — {phase}"

        if log_lines and log_lines[-1].startswith("[Stan]"):
            log_lines[-1] = msg
        else:
            log_lines.append(msg)


def _run_cmdstan(prog, Y, K, draws, tune, chains, lkj_eta, slab_scale, seed,
                 identification="lower_triangular", anchors=None,
                 orthogonal=False, missing_model="woodbury"):
    import sys
    from befa_stan import fit_stan

    prog.phase = "stan"
    prog.total = (draws + tune) * chains
    prog.message = "initializing chains…"
    if orthogonal:
        prog.message += " [orthogonal mode → correlated-factor Stan model]"

    # Stan writes its C-level subprocess stdout directly to fd 1 (the real
    # terminal), completely bypassing Python's sys.stdout object.  The
    # _TeeWriter approach therefore cannot capture Stan's iteration output.
    # Instead we:
    #   (a) Keep show_console=True so the terminal still shows full output.
    #   (b) Tee Python-level prints (from befa_stan.py) via _TeeWriter.
    #   (c) Launch a background thread that polls the chain CSV files Stan
    #       writes to disk and feeds row counts into prog.log_lines.
    log_dir = Path.home() / "stan_logs"
    log_dir.mkdir(exist_ok=True)

    # Snapshot existing CSVs before the run so we can identify new ones
    existing_csvs = set(glob.glob(str(log_dir / "*.csv")))

    stop_poll  = threading.Event()
    poll_thread = threading.Thread(
        target=_poll_stan_csvs,
        args=(log_dir, existing_csvs,
              (draws + tune) * chains,   # total_draws
              tune * chains,             # warmup_draws
              prog.log_lines, stop_poll),
        daemon=True,
    )

    orig_stdout = sys.stdout
    sys.stdout  = _TeeWriter(orig_stdout, prog.log_lines)
    try:
        print(
            f"[fit_runner] CmdStan NUTS | K={K} | chains={chains} | "
            f"warmup={tune} | sampling={draws} | missing_model={missing_model!r}",
            flush=True,
        )
        poll_thread.start()
        fit = fit_stan(
            Y, K=K, chains=chains, iter_warmup=tune, iter_sampling=draws,
            seed=seed, lkj_eta=lkj_eta, slab_scale=slab_scale,
            show_progress=False, show_console=True,
            identification=identification, anchors=anchors,
            missing_model=missing_model,
        )
        print("[fit_runner] Sampling complete.", flush=True)
    finally:
        stop_poll.set()
        sys.stdout = orig_stdout

    poll_thread.join(timeout=3.0)
    prog.iter = prog.total
    return {"stan_fit": fit}


# ---------- Public entry point ----------------------------------------
def start_fit(prog: FitProgress, backend: str, **kwargs):
    """Spawn a daemon thread that runs the chosen backend."""
    prog.status = "running"
    prog.backend = backend
    prog.iter = 0
    prog.total = 0
    prog.message = "starting…"
    prog.error = ""
    prog.result = None
    prog.log_lines = collections.deque(maxlen=_LOG_MAXLEN)
    prog.started_at = time.time()
    prog.finished_at = 0.0
    prog.cancel_event.clear()

    def _target():
        try:
            if backend == "PyMC ADVI (fast)":
                res = _run_pymc_advi(prog, **kwargs)
            elif backend == "PyMC NUTS (full)":
                res = _run_pymc_nuts(prog, **kwargs)
            elif backend == "CmdStan NUTS (full)":
                res = _run_cmdstan(prog, **kwargs)
            else:
                raise ValueError(f"unknown backend {backend}")
            prog.result = res
            prog.status = "done"
            prog.message = "finished"
        except _Cancelled:
            prog.status = "cancelled"
            prog.message = "cancelled by user"
        except Exception as e:
            prog.status = "error"
            prog.error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            prog.message = "error (see Diagnostics)"
        finally:
            prog.finished_at = time.time()

    t = threading.Thread(target=_target, daemon=True)
    prog.thread = t
    t.start()
