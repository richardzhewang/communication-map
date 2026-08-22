"""The general map: coupling coefficient C for all 18 classes + the census.

One code path for every model, GPT-2 to Pythia-6.9B. GPU-first, dimensions
from the model config, statistics shared with experiments/lib/commap (the
CPU reference implementation, kept as a cross-check):

  head-head       factored traces tr(gR (A^T O) gV (A^T O)^T), batched
                  on GPU in fp32 (TF32 off), never materializing [d, d];
                  raw C tables written for Application 1's selection.
  interfaces      the same factored trick against the interface-matrix
                  Gram; shared-rotation z rankings (dense classes).
  mixed classes   quadratic forms, chunked over heads on GPU; C scored
                  and censused, no selection.
  neuron-neuron   one tiled GPU pass: per-span histograms, the wires
                  (|cos| >= 0.5 against the exact chance law), max |cos|.
  census          every head pair standardized against its theoretical
                  rotation null distribution via the closed-form
                  Weingarten moments (verified by verify_census_mc.py).

Selection (empirical null distributions + BH) is NOT done here; it is
Application 1's tooling (app1_select.py) and covers head classes only.

Usage: uv run python experiments/map_build.py MODEL [--n-rot 500]
Writes results/map/{MODEL}/head_C.csv.gz, theory_census.json,
families.csv, nn_hist.npz, wires.csv.gz, summary.json, top_edges_*.csv.gz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "lib"))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)

DEV = "cuda"
WIRE_CUT = 0.5
FAM: list = []          # census rows, appended per family
OUT_DIR: Path = None    # set in main()


# ------------------------------------------------------------------ loading

def load_tl(name: str):
    from transformer_lens import HookedTransformer
    kw = dict(fold_ln=True, center_writing_weights=True, center_unembed=True,
              device=DEV, dtype=torch.float32)
    if name.startswith("pythia"):
        from transformers import AutoModelForCausalLM
        # fp32 preload: with fp16 the TL shim folds in fp16 (validated
        # 2e-4 rel drift). Big models use stream_extract instead.
        hf = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{name}",
                                                  torch_dtype=torch.float32)
        if not hasattr(hf, "embed_out"):
            hf.embed_out = getattr(hf, "lm_head", None) \
                or hf.get_output_embeddings()
        m = HookedTransformer.from_pretrained(name, hf_model=hf, **kw).eval()
        del hf
    else:
        m = HookedTransformer.from_pretrained(name, **kw).eval()
    return m


def extract(model) -> dict:
    """Pull everything the map needs to CPU fp32, then free the model."""
    cfg = model.cfg
    with torch.no_grad():
        d = {
            "L": cfg.n_layers, "H": cfg.n_heads, "d": cfg.d_model,
            "dh": cfg.d_head, "d_mlp": cfg.d_mlp,
            # column-convention factors [L, H, d, dh]
            "Q": model.W_Q.float().cpu(), "K": model.W_K.float().cpu(),
            "V": model.W_V.float().cpu(),
            "O": model.W_O.float().transpose(-1, -2).contiguous().cpu(),
            # neuron reader/writer rows [L, d_mlp, d]
            "Win": model.W_in.float().transpose(-1, -2).contiguous().cpu(),
            "Wout": model.W_out.float().cpu(),
            "H_emb": (model.W_E.float().T @ model.W_E.float()).cpu(),
            "G_unemb": (model.W_U.float() @ model.W_U.float().T).cpu(),
            "learned_pos": cfg.positional_embedding_type
            in ("standard", "shortformer"),
        }
        if d["learned_pos"]:
            d["H_pos"] = (model.W_pos.float().T @ model.W_pos.float()).cpu()
    return d


def stream_extract(name: str) -> dict:
    """Memory-streamed alternative to load_tl+extract for models whose
    fp32 processing copy does not fit RAM alongside the HF copy (6.9B).

    Uses TL's OWN conversion and per-layer processing functions
    (convert_neox_weights, ProcessWeights._fold_layer, and the centering
    ops verbatim), applied layer by layer in fp32 with the fp16 source
    freed as it is consumed. Validated bit-close against the TL path on
    pythia-160m/2.8b by experiments/verify_stream.py.
    """
    import gc
    from transformer_lens.loading_from_pretrained import (
        get_pretrained_model_config)
    from transformer_lens.pretrained.weight_conversions import (
        convert_neox_weights)
    from transformer_lens.weight_processing import ProcessWeights
    from transformers import AutoModelForCausalLM

    cfg = get_pretrained_model_config(name)
    hf = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{name}",
                                              torch_dtype=torch.float16)
    if not hasattr(hf, "embed_out"):
        hf.embed_out = getattr(hf, "lm_head", None) \
            or hf.get_output_embeddings()
    sd = dict(convert_neox_weights(hf, cfg))
    del hf
    gc.collect()
    L, Hh, d = cfg.n_layers, cfg.n_heads, cfg.d_model
    dh, dm = cfg.d_head, cfg.d_mlp
    Q = torch.empty(L, Hh, d, dh)
    K = torch.empty(L, Hh, d, dh)
    V = torch.empty(L, Hh, d, dh)
    O = torch.empty(L, Hh, d, dh)
    Win = torch.empty(L, dm, d)
    Wout = torch.empty(L, dm, d)
    for l in range(L):
        keys = [k for k in sd if k.startswith(f"blocks.{l}.")]
        mini = {k: sd[k].float() for k in keys}
        mini = ProcessWeights._fold_layer(mini, cfg, l, True, True, None, "")
        WO = mini[f"blocks.{l}.attn.W_O"]            # [H, dh, d]
        WO = WO - WO.mean(-1, keepdim=True)          # center writing
        Wo = mini[f"blocks.{l}.mlp.W_out"]           # [dm, d]
        Wo = Wo - Wo.mean(-1, keepdim=True)
        Q[l] = mini[f"blocks.{l}.attn.W_Q"]          # [H, d, dh]
        K[l] = mini[f"blocks.{l}.attn.W_K"]
        V[l] = mini[f"blocks.{l}.attn.W_V"]
        O[l] = WO.transpose(-1, -2)                  # column conv [H, d, dh]
        Win[l] = mini[f"blocks.{l}.mlp.W_in"].T      # readers [dm, d]
        Wout[l] = Wo
        for k in keys:
            del sd[k]
        del mini
        gc.collect()
    end = {k: v.float() for k, v in sd.items()}
    del sd
    end = ProcessWeights._fold_unembed_layer_norm(end, cfg, True, True, None)
    WE = end["embed.W_E"]
    WE = WE - WE.mean(-1, keepdim=True)              # center writing
    WU = end["unembed.W_U"]                          # [d, d_vocab]
    WU = WU - WU.mean(-1, keepdim=True)              # center unembed
    WEg, WUg = WE.to(DEV), WU.to(DEV)
    W = {
        "L": L, "H": Hh, "d": d, "dh": dh, "d_mlp": dm,
        "Q": Q, "K": K, "V": V, "O": O, "Win": Win, "Wout": Wout,
        "H_emb": (WEg.T @ WEg).cpu(), "G_unemb": (WUg @ WUg.T).cpu(),
        "learned_pos": cfg.positional_embedding_type
        in ("standard", "shortformer"),
    }
    del WEg, WUg, end
    torch.cuda.empty_cache()
    return W


def _unit_rows_t(X: torch.Tensor) -> torch.Tensor:
    n = X.norm(dim=-1, keepdim=True)
    return X / torch.clamp(n, min=1e-30)


# ------------------------------------------------------------------ families

def head_head(W, out_rows):
    """K/Q/V-composition via factored traces, all heads batched on GPU.
    Raw C only; selection is Application 1's job."""
    L, Hh, dh = W["L"], W["H"], W["dh"]
    Nh = L * Hh
    layers = np.repeat(np.arange(L), Hh)
    labels = np.array([f"L{l}H{h}" for l in range(L) for h in range(Hh)])
    O = W["O"].reshape(Nh, W["d"], dh).to(DEV)   # kept: reused throughout
    V = W["V"].reshape(Nh, W["d"], dh).to(DEV)   # one upload, Gram, freed
    gV = torch.matmul(V.transpose(1, 2), V).double()
    gO = torch.matmul(O.transpose(1, 2), O).double()
    del V
    tr_Hov = torch.einsum("hij,hij->h", gO, gV).cpu().numpy()  # ||W_OV||_F^2
    r_i, w_i = np.meshgrid(np.arange(Nh), np.arange(Nh), indexing="ij")
    mask = layers[w_i] < layers[r_i]

    for cls, outer_k, inner_k in [("head_head_K", "K", "Q"),
                                  ("head_head_Q", "Q", "K"),
                                  ("head_head_V", "V", "O")]:
        A = W[outer_k].reshape(Nh, W["d"], dh).to(DEV) if outer_k != "O" else O
        Binner = W[inner_k].reshape(Nh, W["d"], dh).to(DEV) \
            if inner_k != "O" else O
        gR_in = torch.matmul(Binner.transpose(1, 2), Binner).double()  # inner
        gR_out = torch.matmul(A.transpose(1, 2), A).double()
        tr_G = torch.einsum("hij,hij->h", gR_out, gR_in).cpu().numpy()
        raw = torch.empty(Nh, Nh, dtype=torch.float64)
        blk = 32
        gR_in32 = gR_in.float()
        O_flat = O.permute(1, 0, 2).reshape(W["d"], Nh * dh)    # [d, Nh*dh]
        for i in range(0, Nh, blk):
            b = min(blk, Nh - i)
            # X[r, w] = A_r^T O_w without broadcast expansion of A:
            # [b, dh, d] @ [d, Nh*dh] -> [b, dh, Nh*dh]
            Xf = torch.matmul(A[i:i + b].transpose(1, 2), O_flat)
            X = Xf.reshape(b, dh, Nh, dh).permute(0, 2, 1, 3).contiguous()
            del Xf
            T = torch.matmul(torch.matmul(X, gV.float().unsqueeze(0)),
                             X.transpose(2, 3))                 # X gV X^T
            raw[i:i + b] = torch.einsum(
                "bij,bwij->bw", gR_in32[i:i + b], T).double().cpu()
            del X, T
        del O_flat
        C = np.sqrt(np.clip(raw.numpy(), 0, None)
                    / (tr_G[:, None] * tr_Hov[None, :]))
        out_rows.append(dict(cls=cls, stat=C[mask],
                             writer=labels[w_i[mask]], reader=labels[r_i[mask]],
                             null_kind="raw"))
        del A, gR_in, gR_out, gR_in32
        torch.cuda.empty_cache()
    return dict(gV=gV, gO=gO, tr_Hov=tr_Hov, labels=labels, layers=layers,
                O=O)


def theory_census(W, head_rows):
    """Every head pair standardized against its theoretical rotation null
    distribution on the C^2 scale: z = (T - E[T]) / SD[T] with
    T = tr(G QHQ^T) = C^2 tr G tr H, both moments exact and closed-form
    (mean tr G tr H/d; variance 2/((d-1)(d+2))
    (trG^2-(trG)^2/d)(trH^2-(trH)^2/d)). No delta method."""
    L, H, d = W["L"], W["H"], W["d"]
    NH = L * H
    fold = lambda X: np.asarray(X, dtype=np.float64).reshape(NH, d, -1)
    Qf, Kf, Vf, Of = fold(W["Q"]), fold(W["K"]), fold(W["V"]), fold(W["O"])
    g = lambda X: np.matmul(X.transpose(0, 2, 1), X)
    gQ, gK, gV, gO = g(Qf), g(Kf), g(Vf), g(Of)

    def inv(a, b):
        ab = np.matmul(a, b)
        return (np.einsum("hii->h", ab),
                np.einsum("hij,hji->h", ab, ab))

    trQK, trQK2 = inv(gQ, gK)
    trOV, trOV2 = inv(gO, gV)
    rinv = {"head_head_K": (trQK, trQK2), "head_head_Q": (trQK, trQK2),
            "head_head_V": (trOV, trOV2)}
    cH = trOV2 - trOV ** 2 / d

    lab = lambda s: (pd.Series(s).str.extract(r"L(\d+)H(\d+)").astype(int)
                     .apply(lambda r: r[0] * H + r[1], axis=1).values)
    summary = {"route": "closed-form Weingarten moments, z on C^2",
               "channels": {}}
    for r in head_rows:
        c = r["cls"]
        trG, trG2 = rinv[c]
        ri, wi = lab(r["reader"]), lab(r["writer"])
        span = ri // H - wi // H
        mu_T = trG[ri] * trOV[wi] / d
        var_T = 2.0 / ((d - 1) * (d + 2)) * (trG2 - trG ** 2 / d)[ri] * cH[wi]
        T = r["stat"] ** 2 * trG[ri] * trOV[wi]
        z = (T - mu_T) / np.sqrt(var_T)
        rows = [dict(span=int(s), n=int(m.sum()),
                     above2=float((z[m] >= 2).mean()),
                     below2=float((z[m] <= -2).mean()),
                     above3=float((z[m] >= 3).mean()),
                     below3=float((z[m] <= -3).mean()),
                     median_z=float(np.median(z[m])))
                for s in np.unique(span) for m in [span == s]]
        a2, b2 = float((z >= 2).mean()), float((z <= -2).mean())
        za, zb = z[z >= 2], z[z <= -2]
        summary["channels"][c] = dict(
            n=int(len(z)), above2=a2, below2=b2, within2=1 - a2 - b2,
            above3=float((z >= 3).mean()), below3=float((z <= -3).mean()),
            median_z=float(np.median(z)),
            median_z_above=float(np.median(za)) if len(za) else None,
            median_z_below=float(np.median(zb)) if len(zb) else None,
            per_span=rows)
    return summary


def interfaces_head(W, hh, n_rot, out_rows, rng):
    """emb/pos_head_{K,Q,V} + head_unembed with shared-rotation nulls.
    Rotations are processed in chunks so the rotated-Gram stack stays
    under a few GB (a full 500-rotation stack at d=4096 would be 33 GB).
    The rng draw order matches the unchunked version exactly."""
    Nh = W["L"] * W["H"]
    dh, d = W["dh"], W["d"]
    labels = hh["labels"]
    rot_chunk = max(1, min(n_rot, int(4e9 / (d * d * 4))))

    def rot_chunk_stack(Hb: torch.Tensor, c: int) -> torch.Tensor:
        """[c, d, d] conjugated Grams (fp32, on GPU)."""
        out = torch.empty(c, d, d, device=DEV)
        for r in range(c):
            g = torch.from_numpy(
                rng.standard_normal((d, d))).float().to(DEV)
            Qr, _ = torch.linalg.qr(g)
            out[r] = Qr @ Hb @ Qr.T
        return out

    def traces_vs(Hb, A, g_in):
        """tr(G_r Hb) for all readers r: tr(g_in (A^T Hb A)). [Nh]"""
        Z = torch.matmul(torch.matmul(A.transpose(1, 2), Hb.unsqueeze(0)), A)
        return torch.einsum("hij,hij->h", g_in.float(), Z).double()

    for wname, key in [("EMB", "H_emb")] + \
            ([("POS", "H_pos")] if W["learned_pos"] else []):
        Hb = W[key].to(DEV)
        tr_Hb = float(torch.einsum("ii->", Hb.double()))
        setups = []
        for cls_slot, outer_k, inner_k in [("K", "K", "Q"), ("Q", "Q", "K"),
                                           ("V", "V", "O")]:
            A = W[outer_k].reshape(Nh, d, dh).to(DEV) if outer_k != "O" \
                else hh["O"]
            Bi = W[inner_k].reshape(Nh, d, dh).to(DEV) if inner_k != "O" \
                else hh["O"]
            g_in = torch.matmul(Bi.transpose(1, 2), Bi)
            tr_G = torch.einsum(
                "hij,hij->h", torch.matmul(A.transpose(1, 2), A).double(),
                g_in.double()).cpu().numpy()
            raw = traces_vs(Hb, A, g_in).cpu().numpy()
            stat = np.sqrt(np.clip(raw, 0, None) / (tr_G * tr_Hb))
            setups.append(dict(slot=cls_slot, A=A, g_in=g_in, tr_G=tr_G,
                               stat=stat, ens=np.empty((n_rot, Nh))))
            del Bi
        done = 0
        while done < n_rot:
            c = min(rot_chunk, n_rot - done)
            rots = rot_chunk_stack(Hb, c)
            for s_ in setups:
                for j in range(c):
                    s_["ens"][done + j] = np.sqrt(np.clip(
                        traces_vs(rots[j], s_["A"], s_["g_in"]).cpu().numpy(),
                        0, None) / (s_["tr_G"] * tr_Hb))
            done += c
            del rots
            torch.cuda.empty_cache()
        for s_ in setups:
            z = (s_["stat"] - s_["ens"].mean(0)) \
                / np.maximum(s_["ens"].std(0), 1e-12)
            out_rows.append(dict(
                cls=f"{wname.lower()}_head_{s_['slot']}", stat=s_["stat"],
                z=z,
                writer=np.full(Nh, wname), reader=labels,
                null_kind="rotation-dense"))
        del Hb, setups
        torch.cuda.empty_cache()

    # head -> unembed: reader Gram G_U conjugated
    Gu = W["G_unemb"].to(DEV)
    tr_Gu = float(torch.einsum("ii->", Gu.double()))
    Ov = hh["O"]
    gV32 = hh["gV"].float()
    raw = torch.einsum("hij,hij->h", gV32, torch.matmul(
        torch.matmul(Ov.transpose(1, 2), Gu.unsqueeze(0)), Ov)).double()
    stat = np.sqrt(np.clip(raw.cpu().numpy(), 0, None)
                   / (hh["tr_Hov"] * tr_Gu))
    ens = np.empty((n_rot, Nh))
    done = 0
    while done < n_rot:
        c = min(rot_chunk, n_rot - done)
        rots = rot_chunk_stack(Gu, c)
        for j in range(c):
            ens[done + j] = np.sqrt(np.clip(
                torch.einsum("hij,hij->h", gV32, torch.matmul(
                    torch.matmul(Ov.transpose(1, 2), rots[j].unsqueeze(0)),
                    Ov)).double().cpu().numpy(), 0, None)
                / (hh["tr_Hov"] * tr_Gu))
        done += c
        del rots
        torch.cuda.empty_cache()
    z = (stat - ens.mean(0)) / np.maximum(ens.std(0), 1e-12)
    out_rows.append(dict(cls="head_unembed", stat=stat, z=z,
                         writer=hh["labels"], reader=np.full(Nh, "UNEMB"),
                         null_kind="rotation-dense"))
    del Gu
    torch.cuda.empty_cache()


def mixed(W, hh, out_rows):
    """head_neuron + neuron_head_{K,Q,V}: C scored and censused, chunked
    over heads on GPU. No selection (Application 1 is head-head only)."""
    L, Hh, d, dh, dm = W["L"], W["H"], W["d"], W["dh"], W["d_mlp"]
    Nh, Nt = L * Hh, L * dm
    h_layer = np.repeat(np.arange(L), Hh)
    n_layer = np.repeat(np.arange(L), dm)

    def small_of(key):
        F = W[key].reshape(Nh, d, dh).to(DEV)
        s = torch.matmul(F.transpose(1, 2), F)
        del F
        return s

    gQ, gK = small_of("Q"), small_of("K")
    trQK = torch.einsum("hij,hij->h", gQ.double(), gK.double()).cpu().numpy()
    gO = torch.matmul(hh["O"].transpose(1, 2), hh["O"]).float()
    cfgs = [
        ("head_neuron", "Win", "O", hh["gV"].float(),
         np.sqrt(hh["tr_Hov"]), True),
        ("neuron_head_K", "Wout", "K", gQ, np.sqrt(trQK), False),
        ("neuron_head_Q", "Wout", "Q", gK, np.sqrt(trQK), False),
        ("neuron_head_V", "Wout", "V", gO, np.sqrt(hh["tr_Hov"]), False),
    ]

    h16 = h_layer.astype(np.int16)
    n16 = n_layer.astype(np.int16)
    for cls, vec_k, proj_k, small, den, to_neuron in cfgs:
        vecs = _unit_rows_t(W[vec_k].reshape(Nt, d)).to(DEV)
        proj = hh["O"] if proj_k == "O" \
            else W[proj_k].reshape(Nh, d, dh).to(DEV)
        stat = np.empty((Nh, Nt), dtype=np.float32)
        blk = max(4, int(2.5e9 / (Nt * dh * 4)))
        for i in range(0, Nh, blk):
            P = proj[i:i + blk]                              # [b, d, dh]
            Xc = torch.matmul(vecs.unsqueeze(0), P)          # [b, Nt, dh]
            Y = torch.matmul(Xc, small[i:i + blk])           # [b, Nt, dh]
            qf = torch.einsum("bnd,bnd->bn", Y, Xc)          # [b, Nt]
            stat[i:i + blk] = (qf.clamp_min(0).sqrt().cpu().numpy()
                               / den[i:i + blk, None])
            del P, Xc, Y, qf
        del vecs
        if proj_k != "O":
            del proj
        torch.cuda.empty_cache()
        if to_neuron:
            emask = h16[:, None] <= n16[None, :]
        else:
            emask = n16[None, :] < h16[:, None]
        s = stat[emask]                                       # fp32
        del stat
        flat = np.flatnonzero(emask.ravel())
        del emask
        order = np.argsort(-s)[:2000]
        tf = flat[order]
        del flat
        hi, ni = tf // Nt, tf % Nt
        head_l = hh["labels"][hi]
        neu_l = np.array([f"L{n_layer[j]}N{j % dm}" for j in ni])
        pd.DataFrame({"writer": head_l if to_neuron else neu_l,
                      "reader": neu_l if to_neuron else head_l,
                      "stat": s[order]}).to_csv(
            OUT_DIR / f"top_edges_{cls}.csv.gz", index=False)
        FAM.append(dict(cls=cls, n_edges=len(s),
                        max_stat=float(s.max()), null_kind="scored"))
        print(f"    {cls}: {len(s)} edges scored", flush=True)
        del s
    del gQ, gK, gO
    torch.cuda.empty_cache()


def neuron_lab(n_layer, ni, dm):
    return np.array([f"L{n_layer[i]}N{i % dm}" for i in ni])


def interfaces_neuron(W, out_rows, rng):
    """emb_neuron (+pos_neuron if learned pos) and neuron_unembed with the
    exact sphere null (eigenvalues + sampled unit vectors)."""
    d, dm, L = W["d"], W["d_mlp"], W["L"]
    Nt = L * dm
    n_layer = np.repeat(np.arange(L), dm)
    ni = np.arange(Nt)
    Rhat = _unit_rows_t(W["Win"].reshape(Nt, d)).to(DEV)
    What = _unit_rows_t(W["Wout"].reshape(Nt, d)).to(DEV)
    cfgs = [("emb_neuron", "H_emb", Rhat)] + \
        ([("pos_neuron", "H_pos", Rhat)] if W["learned_pos"] else []) + \
        [("neuron_unembed", "G_unemb", What)]
    for cls, key, vecs in cfgs:
        Hb = W[key].double().to(DEV)
        tr = float(torch.einsum("ii->", Hb))
        qf = torch.empty(Nt, dtype=torch.float64, device=DEV)
        for i in range(0, Nt, 65536):
            vd = vecs[i:i + 65536].double()
            qf[i:i + 65536] = torch.einsum("nd,nd->n", vd @ Hb, vd)
            del vd
        stat2 = (qf / tr).clamp_min(0).cpu().numpy()
        lam = torch.linalg.eigvalsh(Hb).clamp_min(0)
        gs = torch.from_numpy(
            rng.standard_normal((100_000, d))).float().to(DEV) ** 2
        u2 = gs / gs.sum(1, keepdim=True)
        null = (u2.double() @ (lam / lam.sum())).cpu().numpy()
        z = (stat2 - null.mean()) / null.std()
        out_rows.append(dict(cls=cls, stat=np.sqrt(stat2), z=z,
                             writer=(np.full(Nt, key.split("_")[1].upper())
                                     if cls != "neuron_unembed"
                                     else neuron_lab(n_layer, ni, dm)),
                             reader=(neuron_lab(n_layer, ni, dm)
                                     if cls != "neuron_unembed"
                                     else np.full(Nt, "UNEMB")),
                             null_kind="sphere-dense"))
        del Hb, gs, u2
        torch.cuda.empty_cache()


NN_THRESHOLDS = [0.15, 0.20, 0.23, 0.25, 0.30, 0.50]


def neuron_neuron(W, n_bins=4001):
    """One tiled signed-cosine pass: per-span histograms for the census
    figure, exact exceedance counts at NN_THRESHOLDS (the paper's
    Table 6), the wires (|cos| >= WIRE_CUT), and the maximum |cos|."""
    L, d, dm = W["L"], W["d"], W["d_mlp"]
    Wout = _unit_rows_t(W["Wout"]).to(DEV)   # [L, dm, d]
    Win = _unit_rows_t(W["Win"]).to(DEV)
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    spans = list(range(1, L))
    hist = np.zeros((len(spans), n_bins), dtype=np.int64)
    tiles = [(lw, lr) for lw in range(L) for lr in range(lw + 1, L)]
    wires = []
    max_abs = 0.0
    exceed = {t: 0 for t in NN_THRESHOLDS}
    for lw, lr in tiles:
        c = Wout[lw] @ Win[lr].T
        hist[lr - lw - 1] += torch.histc(c.flatten(), bins=n_bins,
                                         min=-1.0, max=1.0
                                         ).long().cpu().numpy()
        max_abs = max(max_abs, float(c.abs().max()))
        for th in NN_THRESHOLDS:
            exceed[th] += int((c.abs() > th).sum())
        big = c.abs() >= WIRE_CUT
        k = int(big.sum())
        if k:
            idx = big.nonzero(as_tuple=False).cpu().numpy()
            cs = c[big].cpu().numpy()
            wires += [dict(writer=f"L{lw}N{iw}", reader=f"L{lr}N{ir}",
                           stat=float(cv))
                      for (iw, ir), cv in zip(idx, cs)]
        del c, big
    wires = sorted(wires, key=lambda r: -abs(r["stat"]))
    del Wout, Win
    torch.cuda.empty_cache()
    m_total = int(hist.sum())
    return dict(m_total=m_total, hist=hist, edges=edges,
                wires=pd.DataFrame(wires), max_abs=max_abs,
                exceed={str(t): n for t, n in exceed.items()})


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n-rot", type=int, default=500)
    ap.add_argument("--stream-load", action="store_true",
                    help="layer-streamed extraction (for 6.9B RAM limits)")
    args = ap.parse_args()
    global OUT_DIR
    out = REPO / "results" / "map" / args.model
    OUT_DIR = out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()

    print(f"loading {args.model} ...", flush=True)
    import gc
    if args.stream_load:
        W = stream_extract(args.model)
    else:
        model = load_tl(args.model)
        W = extract(model)
        # hard-free: TL holds circular refs, so break storages too
        for p in model.parameters():
            p.data = torch.empty(0)
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  extracted in {time.time()-t0:.0f}s | d={W['d']} L={W['L']} "
          f"H={W['H']} d_mlp={W['d_mlp']} learned_pos={W['learned_pos']}",
          flush=True)

    rows: list[dict] = []
    marks = {}
    t = time.time()
    hh = head_head(W, rows)
    marks["head_head"] = time.time() - t
    print(f"  head_head done {marks['head_head']:.0f}s", flush=True)
    t = time.time()
    census = theory_census(W, rows)
    marks["census"] = time.time() - t
    print(f"  census done {marks['census']:.0f}s", flush=True)
    t = time.time()
    interfaces_head(W, hh, args.n_rot, rows, rng)
    marks["interfaces_head"] = time.time() - t
    print(f"  interfaces_head done {marks['interfaces_head']:.0f}s", flush=True)
    t = time.time()
    mixed(W, hh, rows)
    marks["mixed"] = time.time() - t
    print(f"  mixed done {marks['mixed']:.0f}s", flush=True)
    del hh
    torch.cuda.empty_cache()
    t = time.time()
    interfaces_neuron(W, rows, rng)
    marks["interfaces_neuron"] = time.time() - t
    print(f"  interfaces_neuron done {marks['interfaces_neuron']:.0f}s",
          flush=True)
    t = time.time()
    nn = neuron_neuron(W)
    marks["neuron_neuron"] = time.time() - t
    print(f"  neuron_neuron done {marks['neuron_neuron']:.0f}s", flush=True)

    fam = list(FAM)
    heads_frames = []
    for r in rows:
        fam.append(dict(cls=r["cls"], n_edges=len(r["stat"]),
                        max_stat=float(r["stat"].max()),
                        null_kind=r["null_kind"]))
        if r["cls"].startswith("head_head"):
            heads_frames.append(pd.DataFrame(dict(
                cls=r["cls"], writer=r["writer"], reader=r["reader"],
                stat=r["stat"])))
            continue
        order = np.argsort(-r["stat"])[:2000]
        cols = {k: np.asarray(r[k])[order]
                for k in ("writer", "reader", "stat") + (
                    ("z",) if "z" in r else ())}
        pd.DataFrame(cols).to_csv(
            out / f"top_edges_{r['cls']}.csv.gz", index=False)
    heads_df = pd.concat(heads_frames, ignore_index=True)
    heads_df.to_csv(out / "head_C.csv.gz", index=False)
    fam.append(dict(cls="neuron_neuron", n_edges=nn["m_total"],
                    max_stat=nn["max_abs"], null_kind="exact-law"))
    famdf = pd.DataFrame(fam)
    famdf.to_csv(out / "families.csv", index=False)
    np.savez_compressed(out / "nn_hist.npz",
                        hist=nn["hist"], edges=nn["edges"])
    nn["wires"].to_csv(out / "wires.csv.gz", index=False)
    census["model"] = args.model
    census["d"] = W["d"]
    census["n_wires"] = int(len(nn["wires"]))
    census["nn_max_abs_cos"] = nn["max_abs"]
    census["nn_exceedance"] = nn["exceed"]
    (out / "theory_census.json").write_text(json.dumps(census, indent=2))
    summary = dict(model=args.model, seconds=round(time.time() - t0, 1),
                   marks={k: round(v, 1) for k, v in marks.items()},
                   n_rot=args.n_rot,
                   dims={k: W[k] for k in ("L", "H", "d", "dh", "d_mlp")},
                   learned_pos=W["learned_pos"],
                   total_candidates=int(famdf["n_edges"].sum()),
                   n_wires=int(len(nn["wires"])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(famdf.to_string(index=False))
    print(f"total {summary['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
