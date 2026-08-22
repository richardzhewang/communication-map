"""Spot-check: C_both vs min(C_K, C_Q) for all head-head pairs (l_w < l_r).

C_both := ||M_w^T F_r M_w||_F / (||F_r||_F ||M_w||_F^2)
C_K    := ||F_r M_w||_F   / (||F_r||_F ||M_w||_F)
C_Q    := ||F_r^T M_w||_F / (||F_r||_F ||M_w||_F)

Theorem: C_both <= (sigma_1(M_w)/||M_w||_F) * min(C_K, C_Q) <= min(C_K, C_Q).

All norms via 64x64 identities (factors Q,K,V,O are [d,64]):
  ||F||^2       = tr((Q^T Q)(K^T K))
  ||M||^2       = tr((O^T O)(V^T V))
  ||F M||^2     = tr(A1 Gq A1^T Gv),  A1 := O_w^T K_r
  ||F^T M||^2   = tr(A2 Gk A2^T Gv),  A2 := O_w^T Q_r
  ||M^T F M||^2 = tr(Z^T Gv Z Gv),    Z  := A2 A1^T
"""
import numpy as np
from commap.weights import load_gpt2

w = load_gpt2()
Q = w.Q.reshape(-1, 768, 64).astype(np.float64)  # [144, d, 64]
K = w.K.reshape(-1, 768, 64).astype(np.float64)
V = w.V.reshape(-1, 768, 64).astype(np.float64)
O = w.O.reshape(-1, 768, 64).astype(np.float64)
layers = np.repeat(np.arange(12), 12)

g = lambda X: np.matmul(X.transpose(0, 2, 1), X)          # [144, 64, 64]
Gq, Gk, Gv, Go = g(Q), g(K), g(V), g(O)
normF2 = np.einsum("hij,hji->h", Gq, Gk)                   # ||F_r||^2  [144]
normM2 = np.einsum("hij,hji->h", Go, Gv)                   # ||M_w||^2  [144]

# sigma_1(M_w)^2 = lam_max(Gv^{1/2} Go Gv^{1/2})
def msqrt(Gs):
    vals, vecs = np.linalg.eigh(Gs)
    vals = np.clip(vals, 0, None)
    return vecs @ (np.sqrt(vals)[..., None] * vecs.transpose(0, 2, 1))
sqGv = msqrt(Gv)
s1M = np.sqrt(np.linalg.eigvalsh(sqGv @ Go @ sqGv)[:, -1])  # [144]

Ot = O.transpose(0, 2, 1)                                   # [144, 64, d]
rows = []
for r in range(144):
    lw_mask = layers < layers[r]
    if not lw_mask.any():
        continue
    idx = np.where(lw_mask)[0]
    A1 = Ot[idx] @ K[r]                                     # [n, 64, 64] = O_w^T K_r
    A2 = Ot[idx] @ Q[r]                                     # [n, 64, 64] = O_w^T Q_r
    GvW, s1w, nM2 = Gv[idx], s1M[idx], normM2[idx]
    numK = np.einsum("wij,wji->w", A1 @ Gq[r] @ A1.transpose(0, 2, 1), GvW)
    numQ = np.einsum("wij,wji->w", A2 @ Gk[r] @ A2.transpose(0, 2, 1), GvW)
    Z = A2 @ A1.transpose(0, 2, 1)
    numB = np.einsum("wij,wji->w", Z.transpose(0, 2, 1) @ GvW @ Z, GvW)
    cK = np.sqrt(np.clip(numK, 0, None) / (normF2[r] * nM2))
    cQ = np.sqrt(np.clip(numQ, 0, None) / (normF2[r] * nM2))
    cB = np.sqrt(np.clip(numB, 0, None)) / (np.sqrt(normF2[r]) * nM2)
    for j, wi in enumerate(idx):
        rows.append((wi, r, cK[j], cQ[j], cB[j], s1w[j] / np.sqrt(nM2[j])))

import pandas as pd
df = pd.DataFrame(rows, columns=["w", "r", "cK", "cQ", "cB", "s1_ratio"])
df["min_kq"] = np.minimum(df.cK, df.cQ)
df["ratio"] = df.cB / df.min_kq
df["bound"] = df.s1_ratio * 1.0  # ratio must be <= s1(M)/||M||_F
lab = lambda i: f"L{i // 12}H{i % 12}"

import json
json.dump({"pairs": int(len(df)),
           "bound_violations": int((df.ratio > df.bound + 1e-9).sum()),
           "max_ratio": float(df.ratio.max())},
          open("results/verification/both_sides_check.json", "w"), indent=2)
print(f"pairs: {len(df)}")
print(f"bound violations (ratio > s1/||M||_F + 1e-9): {(df.ratio > df.bound + 1e-9).sum()}")
print(f"ratio quantiles: 50% {df.ratio.quantile(.5):.4f}  90% {df.ratio.quantile(.9):.4f}  "
      f"99% {df.ratio.quantile(.99):.4f}  max {df.ratio.max():.4f}")
print(f"C_both quantiles: 50% {df.cB.quantile(.5):.4f}  99% {df.cB.quantile(.99):.4f}  "
      f"max {df.cB.max():.4f}")
print(f"fraction of slack used, ratio/bound: median {(df.ratio/df.bound).median():.4f}  "
      f"max {(df.ratio/df.bound).max():.4f}")
print("\ntop 10 pairs by ratio (closest to their bound):")
top = df.nlargest(10, "ratio")
for _, t in top.iterrows():
    print(f"  {lab(int(t.w))} -> {lab(int(t.r))}: C_both={t.cB:.4f}  C_K={t.cK:.4f}  "
          f"C_Q={t.cQ:.4f}  ratio={t.ratio:.3f}  bound={t.bound:.3f}")
print("\ntop 10 pairs by C_both absolute:")
for _, t in df.nlargest(10, "cB").iterrows():
    print(f"  {lab(int(t.w))} -> {lab(int(t.r))}: C_both={t.cB:.4f}  C_K={t.cK:.4f}  "
          f"C_Q={t.cQ:.4f}  ratio={t.ratio:.3f}")
