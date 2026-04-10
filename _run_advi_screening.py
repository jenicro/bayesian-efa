import sys
sys.path.insert(0, r'C:\Code\Fun-With-Bayes-EFA')

import numpy as np
import pandas as pd
import pymc as pm
import time

# Load CSV
df = pd.read_csv(r'C:\Users\jenic\Downloads\thinned_data_small.csv').select_dtypes(include=[np.number])
Y = df.values
print(f'Data: {Y.shape}, missing: {np.isnan(Y).sum()} ({np.isnan(Y).mean():.1%})')
print(f'Columns: {list(df.columns[:10])} ... ({df.shape[1]} total)')

from bayesian_efa import build_model
from diagnostics import shrinkage_factor_count

t0 = time.time()
model = build_model(Y, n_factors=6, identification='lower_triangular', orthogonal=True)
print('Model built OK')

with model:
    approx = pm.fit(n=10000, method='advi', progressbar=True, random_seed=1)
    idata = approx.sample(500)

print(f'Done in {time.time()-t0:.1f}s')

Lambda_post = idata.posterior['Lambda'].values
print(f'Lambda_post shape: {Lambda_post.shape}')

Lm = Lambda_post.mean(axis=(0,1))
print('\nPosterior mean loadings (first 10 rows):')
print(np.round(Lm[:10], 2))

print('\nPosterior mean loadings (ALL rows):')
print(np.round(Lm, 2))

# Column SSL
ssl = (Lm**2).sum(axis=0)
print('\nPer-factor SSL:', np.round(ssl, 2))
print('Active factors (SSL>0.5):', (ssl>0.5).sum())
print('Active factors (SSL>1.0):', (ssl>1.0).sum())

# Per-column max loading for a quick sense of factor identity
print('\nMax |loading| per factor (which item drives each):')
for k in range(Lm.shape[1]):
    col = Lm[:, k]
    top_idx = np.abs(col).argmax()
    print(f'  Factor {k+1}: max={col[top_idx]:.3f} at item {top_idx} ({df.columns[top_idx]})')

# Shrinkage diagnostic
Omega_post = idata.posterior.get('Omega', None)
shrink = shrinkage_factor_count(Lambda_post)
print(f'\nShrinkage suggested K: {shrink["suggested_K"]}')
print(f'Per-column P[active]: {np.round(shrink["prob_active"], 2)}')
print(f'Posterior mean K: {shrink["mean_K"]:.2f}')
print(f'Posterior median K: {shrink["median_K"]}')

# ELBO history
hist = approx.hist
if len(hist) > 0:
    print(f'\nFinal ELBO (last 100 mean): {np.mean(hist[-100:]):.1f}')
    print(f'ELBO range: [{hist.min():.1f}, {hist.max():.1f}]')
