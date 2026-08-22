"""Load GPT-2 small weights via TransformerLens and convert to the memo's
column-vector convention, with explicit convention guards (decision D6).

TransformerLens (row convention, activations are row vectors):
    q_i = x_i @ W_Q[l,h]          W_Q: [n_layers, n_heads, d_model, d_head]
    logit_ij  = q_i . k_j / sqrt(d_head)
    head out  = sum_j A_ij (x_j @ W_V[l,h]) @ W_O[l,h]
    logits    = x @ W_U           W_U: [d_model, d_vocab]

Column convention used everywhere downstream (core-research-explainer.tex S2):
    F  := W_QK = Q K^T            [d, d]  bilinear form: logit ~ x_i^T F x_j
          where Q = W_Q[l,h] [d, d_head], K = W_K[l,h] [d, d_head]
    M  := W_OV = O_col V^T        [d, d]  operator: Delta x = M x
          where O_col = W_O[l,h].T [d, d_head], V = W_V[l,h] [d, d_head]
    W_E_col  = W_E.T              [d, |V|]     embedding columns = writers
    W_pos_col = W_pos.T           [d, n_ctx]   positional writers
    W_U_reader = W_U.T            [|V|, d]     unembedding rows = readers

The bilinear form F is the SAME matrix under both conventions
(x_i F x_j^T row == xi^T F xj column); the operator transposes
(row: x M_row -> column: M_row^T x), hence M = (V @ O).T = O_col @ V.T.

Weight processing (D6): fold_ln=True (absorbs LayerNorm gain diag(gamma)C into
the reading weights), center_writing_weights=True (removes the 1-direction from
all writers), center_unembed=True. All arrays converted to float64 numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import D_HEAD, D_MODEL, N_HEADS, N_LAYERS


@dataclass
class Weights:
    """Column-convention weights, float64 numpy.

    Factor matrices are kept (not just the [d,d] products) because both the
    edge statistics and the partialled map (D3) are cheaper on factors.
    """

    Q: np.ndarray      # [L, H, d, d_head]  query-side factor of F
    K: np.ndarray      # [L, H, d, d_head]  key-side factor of F
    V: np.ndarray      # [L, H, d, d_head]  value (input) factor of M
    O: np.ndarray      # [L, H, d, d_head]  output factor of M (column conv)
    W_E: np.ndarray    # [d, n_vocab]       embedding writers
    W_pos: np.ndarray  # [d, n_ctx]         positional writers
    W_U: np.ndarray    # [n_vocab, d]       unembedding readers

    def F(self, l: int, h: int) -> np.ndarray:
        """W_QK for head (l,h): [d, d], rank <= d_head."""
        return self.Q[l, h] @ self.K[l, h].T

    def M(self, l: int, h: int) -> np.ndarray:
        """W_OV for head (l,h): [d, d], rank <= d_head."""
        return self.O[l, h] @ self.V[l, h].T


def load_model(model_name: str = "gpt2"):
    """The processed HookedTransformer (D6 flags). Shared by heads (load_gpt2)
    and neurons (neurons.load_neurons) so the model is only loaded once."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,                  # D6: LayerNorm gain folded into readers
        center_writing_weights=True,   # D6: removes the 1-direction from writers
        center_unembed=True,
        device="cpu",
    )


def load_gpt2(model_name: str = "gpt2", check: bool = True, model=None) -> Weights:
    """Load and convert. Requires network on first call (HF download).
    Pass an existing `model` (from load_model) to avoid re-loading."""
    import torch

    if model is None:
        model = load_model(model_name)
    with torch.no_grad():
        w = Weights(
            Q=model.W_Q.double().numpy(),            # [L, H, d, d_head]
            K=model.W_K.double().numpy(),            # [L, H, d, d_head]
            V=model.W_V.double().numpy(),            # [L, H, d, d_head]
            O=model.W_O.double().numpy().transpose(0, 1, 3, 2),  # -> [L,H,d,d_head]
            W_E=model.W_E.double().numpy().T,        # [d, n_vocab]
            W_pos=model.W_pos.double().numpy().T,    # [d, n_ctx]
            W_U=model.W_U.double().numpy().T,        # [n_vocab, d]
        )

    assert w.Q.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD), w.Q.shape
    assert w.O.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD), w.O.shape

    if check:
        _convention_guard(model, w)
    return w


def _convention_guard(model, w: Weights, l: int = 3, h: int = 7) -> None:
    """Assert our F and M match TransformerLens's own factored circuits.

    TL's model.QK[l,h].AB = W_Q @ W_K^T  (row conv)  == our F      (same matrix)
    TL's model.OV[l,h].AB = W_V @ W_O    (row conv)  == our M^T    (transposed)

    This is the sign/side trap the memo's convention block (S2) exists for;
    fail loudly rather than silently building a transposed map.
    """
    qk = model.QK[l, h].AB.detach().double().numpy()  # [d, d]
    ov = model.OV[l, h].AB.detach().double().numpy()  # [d, d]
    # TL computes AB in float32; a transposed/sided map would mismatch at O(1e-2),
    # so float32-rounding tolerance keeps the guard's full power.
    np.testing.assert_allclose(w.F(l, h), qk, rtol=0, atol=1e-5, err_msg="QK convention mismatch")
    np.testing.assert_allclose(w.M(l, h), ov.T, rtol=0, atol=1e-5, err_msg="OV convention mismatch")
