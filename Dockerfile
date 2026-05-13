# Dockerfile for Fun-With-Bayes-EFA workloads.
#
# The satellite code is NOT baked in — it's mounted at /work at run time so
# iterating on the analysis script doesn't rebuild the image. Only the
# runtime (Python + CmdStan + the deps from requirements.txt) is built in.
#
# Build (from the satellite root):
#   docker build -t europe-west10-docker.pkg.dev/rstudio-486510/analytics/fun-with-bayes-efa:latest .
#
# Smoke test:
#   docker run --rm europe-west10-docker.pkg.dev/rstudio-486510/analytics/fun-with-bayes-efa:latest
#
# Real run (mount the satellite, fit a small simulated BEFA):
#   docker run --rm -v "$PWD:/work" -w /work \
#       europe-west10-docker.pkg.dev/rstudio-486510/analytics/fun-with-bayes-efa:latest \
#       python run_overnight.py simulate --K 3 --N 500 --items-per-factor 5 5 5 \
#           --chains 4 --warmup 1000 --draws 2000 --outdir results/sim_run1
#
# Image lives in Google Artifact Registry; the agent-cloud-runner skill
# (`~/.claude/skills/agent-cloud-runner/SKILL.md`) operates this image
# on a Compute Engine VM.

# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm

RUN apt-get update -qq \
 && apt-get install -yqq --no-install-recommends \
        build-essential gcc g++ gfortran make git curl ca-certificates \
        libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

ENV CMDSTAN_VERSION=2.36.0
ENV CMDSTAN=/opt/cmdstan/cmdstan-${CMDSTAN_VERSION}
ENV PATH="${CMDSTAN}/bin:${PATH}"

WORKDIR /tmp/build
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir cmdstanpy pytest

RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(dir='/opt/cmdstan', version='${CMDSTAN_VERSION}', cores=$(nproc), verbose=True)"

WORKDIR /work

CMD ["python", "-c", "import cmdstanpy, pymc, arviz, numpy, pandas; print({'cmdstanpy': cmdstanpy.__version__, 'cmdstan': cmdstanpy.cmdstan_path(), 'pymc': pymc.__version__, 'arviz': arviz.__version__, 'numpy': numpy.__version__, 'pandas': pandas.__version__})"]
