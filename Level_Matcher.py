import numpy as np, pandas as pd
from lightgbm import LGBMRanker
from scipy.optimize import linear_sum_assignment

# pairs: DataFrame with columns [i, j, z, dE, Lc, Jc, parity_ok, dJ, I_sim, G_jacc, dens_sim, prior, group_i]
features = ["z","dE","Lc","Jc","parity_ok","dJ","I_sim","G_jacc","dens_sim","prior"]

ranker = LGBMRanker(
    objective="lambdarank",
    n_estimators=300, learning_rate=0.05, max_depth=5,
    num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=0
)
group_sizes = pairs.groupby("i").size().tolist()
ranker.fit(pairs[features], pairs["label"], group=group_sizes)

# Inference on new candidate table new_pairs with same feature columns, plus i, j
s = ranker.predict(new_pairs[features])

# Build a cost matrix and assign
I = sorted(new_pairs["i"].unique()); J = sorted(new_pairs["j"].unique())
map_i = {u:k for k,u in enumerate(I)}; map_j = {v:k for k,v in enumerate(J)}
C = np.full((len(I), len(J)), 1e6, float)

for row, sc in zip(new_pairs.itertuples(index=False), s):
    if row.z > 4.0 or row.Lc < 0 or row.Jc < 0:  # hard masks
        continue
    C[map_i[row.i], map_j[row.j]] = -np.log(sc + 1e-6)

ia, ib = linear_sum_assignment(C)
matches = [(I[a], J[b], float(np.exp(-C[a,b]))) for a, b in zip(ia, ib) if C[a,b] < 1e5]
