# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataset import CCCGraphDataset
from cell_morph_features import compute_cell_morph_features
from causal import (
    load_prov_gigapath_tile_encoder,
    HEEncoderWrapper,
    CausalGraphBuilder,
)

# =========================================================
# Optional tqdm
# =========================================================
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cfg_get_int(d: Dict[str, Any], key: str, default: int) -> int:
    """Config int getter that treats explicit null/None as missing."""
    v = d.get(key, default)
    if v is None:
        v = default
    if isinstance(v, str) and v.strip() == "":
        v = default
    return int(v)

def cfg_get_float(d: Dict[str, Any], key: str, default: float) -> float:
    v = d.get(key, default)
    if v is None:
        v = default
    if isinstance(v, str) and v.strip() == "":
        v = default
    return float(v)


def graph_as_tensors(graph_obj: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(graph_obj, dict):
        return graph_obj["edge_index"], graph_obj["edge_attr"]
    return graph_obj.edge_index, graph_obj.edge_attr


def gpu_mem_str() -> str:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        return f"cuda alloc={a:.2f}GB reserved={r:.2f}GB"
    return "cuda n/a"


def np_stats(arr: np.ndarray, name: str, percentiles=(0, 1, 5, 25, 50, 75, 95, 99, 100)) -> Dict[str, Any]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"name": name, "size": 0}
    ps = np.percentile(arr, list(percentiles)).tolist()
    return {
        "name": name,
        "size": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "percentiles": {str(p): float(v) for p, v in zip(percentiles, ps)},
    }


def infer_dataset_id_from_root(root_dir: str | Path) -> str:
    p = Path(root_dir)
    return p.name


# =========================================================
# Pure torch scatter ops (no torch_scatter)
# =========================================================
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out

def scatter_mean_std(src: torch.Tensor, index: torch.Tensor, dim_size: int, eps: float = 1e-6):
    """
    Compute grouped mean/std for src grouped by index.
    src:   [E,H]
    index: [E] group id (dst)
    return mean,std: [dim_size,H]
    """
    E, H = src.shape
    device, dtype = src.device, src.dtype

    ones = torch.ones((E, 1), device=device, dtype=dtype)
    cnt = torch.zeros((dim_size, 1), device=device, dtype=dtype)
    cnt.index_add_(0, index, ones)  # [dim_size,1]
    cnt = cnt.clamp_min(1.0)

    s1 = torch.zeros((dim_size, H), device=device, dtype=dtype)
    s2 = torch.zeros((dim_size, H), device=device, dtype=dtype)
    s1.index_add_(0, index, src)
    s2.index_add_(0, index, src * src)

    mean = s1 / cnt
    var = (s2 / cnt) - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var + eps)
    return mean, std

def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Grouped softmax over index (dst node).
    logits: [E,H] or [E]
    index:  [E] group id (dst)
    """
    if logits.ndim == 1:
        logits_ = logits.unsqueeze(-1)
        squeeze = True
    else:
        logits_ = logits
        squeeze = False

    E, H = logits_.shape
    device = logits_.device
    dtype = logits_.dtype

    if hasattr(torch.Tensor, "scatter_reduce_"):
        max_per = torch.full((dim_size, H), -float("inf"), device=device, dtype=dtype)
        max_per.scatter_reduce_(
            0,
            index.view(-1, 1).expand(-1, H),
            logits_,
            reduce="amax",
            include_self=True,
        )
    else:
        max_per = torch.full((dim_size, H), -float("inf"), device=device, dtype=dtype)
        for i in range(E):
            g = int(index[i].item())
            max_per[g] = torch.maximum(max_per[g], logits_[i])

    exp = torch.exp(logits_ - max_per[index])
    denom = torch.zeros((dim_size, H), device=device, dtype=dtype)
    denom.index_add_(0, index, exp)
    out = exp / (denom[index] + 1e-12)
    return out.squeeze(-1) if squeeze else out


# =========================================================
# Model: Edge-aware attention (Transformer-like, keep name e_{i,j})
# =========================================================
class EdgeAwareAttnLayer(nn.Module):
    """
    Transformer-like + edge bias:

      e_ij^m = ( (WQ x_dst)^T (WK x_src) ) / sqrt(dk)  +  (w_b^m)^T φ(F_ij)
      α_ij^m = softmax_{src in N(dst)}(e_ij^m)         # 训练/传播用 (softmax 后)

    导出强度 strength：使用 e_{i,j} (pre-softmax)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        lr_vocab: int,
        lr_embed_dim: int = 32,
        edge_mlp_hidden: int = 64,
        edge_bias_dim: int = 32,
        dropout: float = 0.2,
        use_headwise_causal_gamma: bool = True,
    ):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.dk = out_dim // heads
        self.out_dim = out_dim
        self.dropout = float(dropout)

        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)

        self.lr_emb = nn.Embedding(lr_vocab, lr_embed_dim)

        fij_dim = 6 + lr_embed_dim
        self.phi = nn.Sequential(
            nn.Linear(fij_dim, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, edge_bias_dim),
            nn.ReLU(inplace=True),
        )

        self.wb = nn.Parameter(torch.empty(heads, edge_bias_dim))
        nn.init.xavier_uniform_(self.wb)

        self.proj = nn.Linear(out_dim, out_dim, bias=True)

        if use_headwise_causal_gamma:
            self.causal_gamma = nn.Parameter(torch.zeros(heads))  # [h]
            self._gamma_headwise = True
        else:
            self.causal_gamma = nn.Parameter(torch.tensor(0.0))   # scalar
            self._gamma_headwise = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr7: torch.Tensor,
        dist_sigma: float,
        coexpr_log1p: bool,
        coexpr_scale: float,
        temperature: float,
        return_logits: bool = False,
    ):
        N = x.size(0)
        src = edge_index[0].long()
        dst = edge_index[1].long()

        q = self.Wq(x).view(N, self.heads, self.dk)
        k = self.Wk(x).view(N, self.heads, self.dk)
        v = self.Wv(x).view(N, self.heads, self.dk)

        qj = q[dst]  # [E,h,dk]
        ki = k[src]  # [E,h,dk]
        vi = v[src]  # [E,h,dk]

        dot = (qj * ki).sum(dim=-1) / math.sqrt(self.dk)  # [E,h]

        tau = float(temperature) if temperature is not None else 1.0
        tau = max(tau, 1e-6)
        dot = dot / tau

        dist = edge_attr7[:, 0].float()
        coexpr = edge_attr7[:, 1].float()
        lr_id = edge_attr7[:, 2].long()
        cmi4 = edge_attr7[:, 3:7].float()  # [E,4] already scaled+normalized in encoder.compose_edge_attr7

        sigma = max(float(dist_sigma), 1e-6)
        dist_decay = torch.exp(-dist / sigma)

        if coexpr_log1p:
            coexpr_norm = torch.log1p(torch.clamp(coexpr, min=0.0)) / max(float(coexpr_scale), 1e-6)
        else:
            coexpr_norm = coexpr / max(float(coexpr_scale), 1e-6)

        cmi4_norm = torch.clamp(cmi4, 0.0, 1.0)

        cont = torch.cat([dist_decay.view(-1,1), coexpr_norm.view(-1,1), cmi4_norm], dim=1)  # [E,6]
        lr_vec = self.lr_emb(lr_id)
        Fij = torch.cat([cont, lr_vec], dim=1)

        phi = self.phi(Fij)
        bias = phi @ self.wb.t()

        e_raw = dot + bias  # [E,h] pre-norm
        # === Scheme B: grouped z-score (per-dst neighborhood) -> tanh ===
        mu_g, std_g = scatter_mean_std(e_raw, dst, dim_size=N, eps=1e-6)  # [N,h], [N,h]
        e_norm = (e_raw - mu_g[dst]) / (std_g[dst] + 1e-6)               # [E,h]
        e = torch.tanh(e_norm)                                           # [E,h] in (-1,1)

        # 训练/传播用 softmax 后的 alpha
        alpha = scatter_softmax(e, dst, dim_size=N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        m = alpha.unsqueeze(-1) * vi
        m = m.reshape(m.size(0), -1)
        out = scatter_sum(m, dst, dim_size=N)

        out = self.proj(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return (out, e) if return_logits else (out, None)


class EdgeAwareEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        heads: int,
        lr_vocab: int,
        lr_embed_dim: int,
        edge_mlp_hidden: int,
        edge_bias_dim: int,
        dropout: float,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            EdgeAwareAttnLayer(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                heads=heads,
                lr_vocab=lr_vocab,
                lr_embed_dim=lr_embed_dim,
                edge_mlp_hidden=edge_mlp_hidden,
                edge_bias_dim=edge_bias_dim,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Learnable per-CMI scaling (applied before concatenating to edge features):
        # edge_attr7 = [dist, coexpr, lr_id, α·I1, β·I2, γ·I3, δ·I4], with each term mapped to ~[0,1].
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.cmi_norm_eps = 1e-6

    def compose_edge_attr7(
        self,
        edge_attr3: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
        I3: torch.Tensor,
        I4: torch.Tensor,
    ) -> torch.Tensor:
        """Return edge_attr7 = [distance_px, coexpr, lr_id, cmi1, cmi2, cmi3, cmi4].

        Each CMI is first scaled by a learnable scalar (α/β/γ/δ), then normalized to ~[0,1]
        via sigmoid(z-score) for stability.
        """
        def _norm01(x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            mu = x.mean()
            std = x.std(unbiased=False)
            return torch.sigmoid((x - mu) / (std + self.cmi_norm_eps)).view(-1, 1)

        c1 = _norm01(self.alpha * I1)
        c2 = _norm01(self.beta  * I2)
        c3 = _norm01(self.gamma * I3)
        c4 = _norm01(self.delta * I4)
        return torch.cat([edge_attr3, c1, c2, c3, c4], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr7: torch.Tensor,
        dist_sigma: float,
        coexpr_log1p: bool,
        coexpr_scale: float,
        temperature: float,
        return_last_logits: bool = False,
    ):
        e_last = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            x, e = layer(
                x, edge_index, edge_attr7,
                dist_sigma=dist_sigma,
                coexpr_log1p=coexpr_log1p,
                coexpr_scale=coexpr_scale,
                temperature=temperature,
                return_logits=(return_last_logits and is_last),
            )
            x = F.elu(x)
            if return_last_logits and is_last:
                e_last = e
        return x, e_last


# =========================================================
# NEW: Standard masked-edge reconstruction with an EDGE DECODER
# =========================================================
class EdgeDecoder(nn.Module):
    """
    Decode edge weight/logit from node embeddings + (non-leaky) edge features.

    为避免“抄答案”的泄露：decoder 默认只用 dist + lr_id（不直接用 coexpr/causal）。
    你仍然可以在 config 里打开 use_coexpr/use_causal，但我不建议在 masked 重建里开。
    """
    def __init__(
        self,
        node_dim: int,
        lr_vocab: int,
        lr_embed_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.2,
        use_coexpr: bool = False,
        use_causal: bool = False,
    ):
        super().__init__()
        self.use_coexpr = bool(use_coexpr)
        self.use_causal = bool(use_causal)

        self.lr_emb = nn.Embedding(lr_vocab, lr_embed_dim)

        feat_dim = 1 + lr_embed_dim
        if self.use_coexpr:
            feat_dim += 1
        if self.use_causal:
            feat_dim += 4

        in_dim = node_dim * 4 + feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_attr7: torch.Tensor) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()

        zs = z[src]
        zd = z[dst]

        feat = [zs, zd, zs * zd, torch.abs(zs - zd)]

        dist = edge_attr7[:, 0].float().unsqueeze(1)
        lr_id = edge_attr7[:, 2].long()
        lr_vec = self.lr_emb(lr_id)

        edge_feat = [dist, lr_vec]
        if self.use_coexpr:
            edge_feat.append(edge_attr7[:, 1].float().unsqueeze(1))
        if self.use_causal:
            edge_feat.append(edge_attr7[:, 3:7].float())

        x = torch.cat(feat + edge_feat, dim=1)
        logit = self.mlp(x).view(-1)
        return logit


# =========================================================
# Diagnostics helpers
# =========================================================
def degree_stats(edge_index: np.ndarray, N: int) -> Dict[str, Any]:
    src = edge_index[0]
    dst = edge_index[1]
    out_deg = np.bincount(src, minlength=N)
    in_deg = np.bincount(dst, minlength=N)

    def _summ(deg, name):
        return {
            "name": name,
            "min": int(deg.min()),
            "max": int(deg.max()),
            "mean": float(deg.mean()),
            "median": float(np.median(deg)),
            "p0": int(np.percentile(deg, 0)),
            "p25": int(np.percentile(deg, 25)),
            "p50": int(np.percentile(deg, 50)),
            "p75": int(np.percentile(deg, 75)),
            "p95": int(np.percentile(deg, 95)),
            "p99": int(np.percentile(deg, 99)),
            "zero_frac": float((deg == 0).mean()),
        }

    return {"out_degree": _summ(out_deg, "out_degree"), "in_degree": _summ(in_deg, "in_degree")}


def build_lr_meta_map(pairs_kept: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for d in pairs_kept:
        pid = int(d["pair_id"])
        out[pid] = {
            "ligand": d.get("ligand", ""),
            "receptor": d.get("receptor", ""),
            "pathway": d.get("pathway", ""),
            "mechanism": d.get("mechanism", ""),
        }
    return out


def global_topk_mask(
    strength: np.ndarray,
    keep_top_percent: float,
    min_keep: int,
    max_keep: int,
    use_min_keep: bool,
) -> np.ndarray:
    """Global top-percent edge filter (CellNEST-style).

    Keep the top `keep_top_percent`% edges by `strength` over *all* edges.
    """
    strength = np.asarray(strength)
    E = int(strength.shape[0])
    keep = np.zeros((E,), dtype=bool)
    if E == 0:
        return keep

    k = int(math.ceil(E * (float(keep_top_percent) / 100.0)))

    if use_min_keep:
        k = max(int(min_keep), k)
    else:
        k = max(1, k)

    k = min(int(max_keep), k, E)

    topk = np.argpartition(-strength, kth=k - 1)[:k]
    keep[topk] = True
    return keep


def aggregate_strength(strength_runs: np.ndarray, method: str) -> np.ndarray:
    method = str(method).lower()
    if method == "mean":
        return strength_runs.mean(axis=0)
    if method == "median":
        return np.median(strength_runs, axis=0)
    if method == "rank_mean":
        R, E = strength_runs.shape
        ranks = np.zeros((R, E), dtype=np.float32)
        for r in range(R):
            order = np.argsort(-strength_runs[r])
            rr = np.empty(E, dtype=np.int64)
            rr[order] = np.arange(E)
            ranks[r] = rr
        mean_rank = ranks.mean(axis=0)
        return 1.0 / (1.0 + mean_rank)
    raise ValueError(f"Unknown ensemble.agg: {method}")


def minmax_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if (mx - mn) <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


# =========================================================
# NEW: teacher weight (w0) for self-supervision
# =========================================================
def build_teacher_w0(
    ea7: torch.Tensor,
    dist_sigma: float,
    coexpr_log1p: bool,
    coexpr_scale: float,
) -> torch.Tensor:
    """
    w0 in [0,1], used as "teacher" target.

    一个稳妥选择：w0 = dist_decay * coexpr_norm * mean(CMI_1..4)
    """
    dist = ea7[:, 0].float()
    coexpr = ea7[:, 1].float()
    cmi4 = ea7[:, 3:7].float()

    sigma = max(float(dist_sigma), 1e-6)
    dist_decay = torch.exp(-dist / sigma)

    if coexpr_log1p:
        coexpr_norm = torch.log1p(torch.clamp(coexpr, min=0.0)) / max(float(coexpr_scale), 1e-6)
    else:
        coexpr_norm = coexpr / max(float(coexpr_scale), 1e-6)

    coexpr_norm = torch.clamp(coexpr_norm, 0.0, 1.0)
    cmi_mean = torch.clamp(cmi4, 0.0, 1.0).mean(dim=1)

    w0 = dist_decay * coexpr_norm * cmi_mean
    w0 = torch.clamp(w0, 0.0, 1.0)
    return w0


def sample_mask_edges(E: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    mask_ratio = float(mask_ratio)
    mask_ratio = min(max(mask_ratio, 0.0), 0.95)
    m = int(math.ceil(E * mask_ratio))
    if m <= 0:
        return torch.zeros((E,), dtype=torch.bool, device=device)
    idx = torch.randperm(E, device=device)[:m]
    mask = torch.zeros((E,), dtype=torch.bool, device=device)
    mask[idx] = True
    return mask


def make_subgraph(edge_index: torch.Tensor, edge_attr: torch.Tensor, keep_mask: torch.Tensor):
    idx = torch.where(keep_mask)[0]
    return edge_index[:, idx], edge_attr[idx], idx


def bpr_ranking_loss(pos_logit: torch.Tensor, neg_logit: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(pos_logit - neg_logit)).mean()


def sample_rank_pairs_same_src(
    edge_index: torch.Tensor,
    teacher_w0: torch.Tensor,
    num_pairs: int,
    device: torch.device,
    max_tries: int = 10,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    从同一 src 的两条边里采样 (e_pos, e_neg)，依据 teacher_w0 排序。
    返回两个 index 张量：pos_idx, neg_idx (shape [K])
    """
    E = int(edge_index.size(1))
    if E <= 1 or num_pairs <= 0:
        return None, None

    src = edge_index[0].long()
    w0 = teacher_w0

    pos_list = []
    neg_list = []

    need = num_pairs
    for _ in range(max_tries):
        e1 = torch.randint(0, E, (need * 2,), device=device)
        e2 = torch.randint(0, E, (need * 2,), device=device)

        same = (src[e1] == src[e2]) & (e1 != e2)
        if same.sum() == 0:
            continue

        e1 = e1[same]
        e2 = e2[same]

        better = w0[e1] > w0[e2]
        worse = w0[e1] < w0[e2]
        valid = better | worse
        if valid.sum() == 0:
            continue

        e1v = e1[valid]
        e2v = e2[valid]
        betterv = better[valid]

        pos = torch.where(betterv, e1v, e2v)
        neg = torch.where(betterv, e2v, e1v)

        pos_list.append(pos)
        neg_list.append(neg)

        got = sum(x.numel() for x in pos_list)
        if got >= num_pairs:
            break

        need = num_pairs - got

    if len(pos_list) == 0:
        return None, None

    pos_idx = torch.cat(pos_list, dim=0)[:num_pairs]
    neg_idx = torch.cat(neg_list, dim=0)[:num_pairs]
    return pos_idx, neg_idx


# =========================================================
# Node encoders (HE + ST) for node features (replace one-hot)
# =========================================================

def _crop_square_rgb_np(he: np.ndarray, cx: float, cy: float, ps: int) -> np.ndarray:
    """Crop RGB square patch centered at (cx,cy). Pad with reflect if out of bounds."""
    H, W, C = he.shape
    assert C == 3, f"Expect HE RGB [H,W,3], got {he.shape}"
    half = ps // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + ps
    y1 = y0 + ps

    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - W); pad_b = max(0, y1 - H)

    if pad_l or pad_t or pad_r or pad_b:
        he_pad = np.pad(he, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="reflect")
        x0 += pad_l; x1 += pad_l
        y0 += pad_t; y1 += pad_t
    else:
        he_pad = he
    return he_pad[y0:y1, x0:x1, :]


def _crop_square_mask_np(mask: np.ndarray, cx: float, cy: float, ps: int) -> np.ndarray:
    """Crop mask square patch centered at (cx,cy). Pad with 0 if out of bounds."""
    H, W = mask.shape
    half = ps // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + ps
    y1 = y0 + ps

    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - W); pad_b = max(0, y1 - H)

    if pad_l or pad_t or pad_r or pad_b:
        m_pad = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0)
        x0 += pad_l; x1 += pad_l
        y0 += pad_t; y1 += pad_t
    else:
        m_pad = mask
    return m_pad[y0:y1, x0:x1]


def build_cell_he_patches(
    he: np.ndarray,
    mask: Optional[np.ndarray],
    coords_xy: np.ndarray,
    cell_ids: Optional[np.ndarray],
    patch_size_px: int,
    mode: str = "square",
) -> np.ndarray:
    """
    Build per-cell HE patches as uint8 [N, ps, ps, 3].

    mode:
      - "square" (NEW default): pure square crop centered at centroid (no mask).
      - "masked_cell"        : keep only the segmented cell area inside the crop.
    """
    ps = int(patch_size_px)
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    N = int(coords_xy.shape[0])
    patches = np.zeros((N, ps, ps, 3), dtype=np.uint8)

    if he.dtype != np.uint8:
        he_u8 = np.clip(he * 255.0, 0, 255).astype(np.uint8) if np.issubdtype(he.dtype, np.floating) else he.astype(np.uint8)
    else:
        he_u8 = he

    mode = str(mode).lower().strip()
    if mode not in {"square", "masked_cell"}:
        raise ValueError(f"Unknown build_cell_he_patches mode: {mode}")

    mask_np = None
    is_binary_like = True
    if mode == "masked_cell":
        if mask is None:
            raise ValueError("mode='masked_cell' requires a segmentation mask")
        mask_np = mask
        if mask_np.dtype not in (np.int32, np.int64, np.uint16, np.uint8):
            mask_np = mask_np.astype(np.int32)
        uniq = np.unique(mask_np)
        is_binary_like = (uniq.size <= 3) and (set(map(int, uniq.tolist())) <= {0, 1, 255})

    for i in range(N):
        cx, cy = float(coords_xy[i, 0]), float(coords_xy[i, 1])
        rgb = _crop_square_rgb_np(he_u8, cx, cy, ps)
        if mode == "square":
            patches[i] = rgb
        else:
            msk = _crop_square_mask_np(mask_np, cx, cy, ps)

            if is_binary_like:
                keep = (msk > 0)
            else:
                lbl = None
                if cell_ids is not None:
                    cid = int(cell_ids[i])
                    if (msk == cid).any():
                        lbl = cid
                if lbl is None:
                    cand = i + 1
                    if (msk == cand).any():
                        lbl = cand
                if lbl is None:
                    keep = (msk > 0)
                else:
                    keep = (msk == lbl)

            out = rgb.copy()
            out[~keep] = 0
            patches[i] = out

    return patches


class STGeneMLP(nn.Module):
    """MLP to embed per-cell gene expression vector -> low-dim node ST embedding."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        in_dim = int(in_dim)
        hidden = int(hidden)
        out_dim = int(out_dim)
        num_layers = int(max(1, num_layers))

        layers: List[nn.Module] = []
        d0 = in_dim
        if num_layers == 1:
            layers.append(nn.Linear(d0, out_dim))
        else:
            layers.append(nn.Linear(d0, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeFeatureEncoder(nn.Module):
    """
    Produce node features by:
      HE patch -> pretrained encoder embedding (optional finetune)
      ST expr  -> MLP embedding (trainable)
      fusion   -> concat
      optional -> concat coords (controlled by cfg['model']['add_coords'])
    """
    def __init__(
        self,
        *,
        gene_dim: int,
        st_hidden: int,
        st_out_dim: int,
        st_num_layers: int,
        st_dropout: float,
        he_hf_model_id: str,
        he_timm_prefix: str,
        he_input_size: int,
        he_out_dim: int,
        he_patch_size_px: int,
        he_batch_size: int,
        he_finetune: bool,
        he_normalize_imagenet: bool,
        precompute_he_if_frozen: bool,
        device: torch.device,
    ):
        super().__init__()
        self.device = device

        self.he_patch_size_px = int(he_patch_size_px)
        self.he_batch_size = int(he_batch_size)
        self.he_out_dim = int(he_out_dim)
        self.he_finetune = bool(he_finetune)
        self.precompute_he_if_frozen = bool(precompute_he_if_frozen)

        self.st_mlp = STGeneMLP(
            in_dim=gene_dim,
            hidden=st_hidden,
            out_dim=st_out_dim,
            num_layers=st_num_layers,
            dropout=st_dropout,
        )

        enc = load_prov_gigapath_tile_encoder(
            hf_model_id=he_hf_model_id,
            timm_prefix=he_timm_prefix,
            device=str(device),
            preflight_download=True,
        )
        self.he_encoder = HEEncoderWrapper(
            encoder=enc,
            input_size=int(he_input_size),
            finetune=bool(he_finetune),
            normalize_imagenet=bool(he_normalize_imagenet),
        )

    @torch.no_grad()
    def precompute_he_embedding(self, he_patches_u8: torch.Tensor) -> torch.Tensor:
        """Precompute HE embedding on current device, return CPU tensor [N, D] float16. Only valid when he_finetune=False."""
        if self.he_finetune:
            raise RuntimeError("precompute_he_embedding requires he_finetune=False")
        self.he_encoder.eval()

        N = int(he_patches_u8.size(0))
        outs: List[torch.Tensor] = []
        bs = max(1, int(self.he_batch_size))
        for i in range(0, N, bs):
            xb = he_patches_u8[i:i+bs].to(self.device, non_blocking=True)
            emb = self.he_encoder(xb, amp=True)
            outs.append(emb.detach().to("cpu", dtype=torch.float16))
        return torch.cat(outs, dim=0)

    def _embed_he(self, he_patches_u8: torch.Tensor) -> torch.Tensor:
        """Return HE embedding [N, D] on device; with grad if finetune."""
        self.he_encoder.train(self.he_finetune)
        N = int(he_patches_u8.size(0))
        outs: List[torch.Tensor] = []
        bs = max(1, int(self.he_batch_size))
        for i in range(0, N, bs):
            xb = he_patches_u8[i:i+bs].to(self.device, non_blocking=True)
            if self.he_finetune:
                emb = self.he_encoder(xb, amp=True)  # finetune with AMP to save memory
            else:
                emb = self.he_encoder(xb, amp=True)
            outs.append(emb)
        return torch.cat(outs, dim=0)

    def _embed_st(self, expr_log_cpu: torch.Tensor, st_batch_size: int = 256) -> torch.Tensor:
        """Embed ST gene vectors in batches to avoid moving full [N,G] to GPU."""
        self.st_mlp.train(True)
        N = int(expr_log_cpu.size(0))
        outs: List[torch.Tensor] = []
        bs = max(1, int(st_batch_size))
        for i in range(0, N, bs):
            xb = expr_log_cpu[i:i+bs].to(self.device, non_blocking=True)
            outs.append(self.st_mlp(xb))
        return torch.cat(outs, dim=0)

    def forward(
        self,
        *,
        expr_log_cpu: torch.Tensor,
        coords_cpu: torch.Tensor,
        add_coords: bool,
        he_patches_u8: Optional[torch.Tensor] = None,
        he_emb_cpu: Optional[torch.Tensor] = None,
        morph_feat_cpu: Optional[torch.Tensor] = None,
        st_batch_size: int = 256,
    ) -> torch.Tensor:
        st_emb = self._embed_st(expr_log_cpu, st_batch_size=st_batch_size)

        if he_emb_cpu is not None:
            he_emb = he_emb_cpu.to(self.device, non_blocking=True).float()
        else:
            if he_patches_u8 is None:
                raise ValueError("Need he_patches_u8 when he_emb_cpu is None")
            he_emb = self._embed_he(he_patches_u8)
            if he_emb.ndim == 3:
                he_emb = he_emb.mean(dim=1)

        if morph_feat_cpu is not None:
            mf = morph_feat_cpu.to(self.device, non_blocking=True).float()
            he_emb = torch.cat([he_emb, mf], dim=1)

        x = torch.cat([he_emb, st_emb], dim=1)

        if add_coords:
            x = torch.cat([x, coords_cpu.to(self.device, non_blocking=True).float()], dim=1)
        return x


# =========================================================
# Build graph once
# =========================================================
def build_graph_and_features(cfg: Dict[str, Any], device: torch.device, proj_device: str):
    dcfg = cfg["dataset"]
    ds = CCCGraphDataset(
        root_dir=dcfg["root"],
        sample_id=dcfg.get("sample_id", None),
        sample_ids=dcfg.get("sample_ids", None),
        lr_pairs_path=dcfg.get("lr_pairs_path", None),
        mpp=float(dcfg.get("mpp", 0.5)),
        neighborhood_um=float(dcfg.get("neighborhood_um", 300.0)),
        neighborhood_px=dcfg.get("neighborhood_px", None),
        quantile_norm=bool(dcfg.get("quantile_norm", True)),
        active_percentile=float(dcfg.get("active_percentile", 98.0)),
        cache_graph=bool(dcfg.get("cache_graph", True)),
        device=None,
    )
    sample_idx = int(dcfg.get("sample_idx", 0))
    if dcfg.get("sample_id", None) is not None:
        sample_idx = 0
    sample = ds[sample_idx]
    sample_id = str(sample["sample_id"])

    # =========================================================
    # (0) Traditional cell morphology features from mask
    # =========================================================
    # mask is expected to be an instance-labeled image where label == cell_id
    mask_np = np.asarray(sample["mask"])
    cell_ids = sample.get("cell_ids", [])
    morph_np, morph_names = compute_cell_morph_features(mask_np, cell_ids, standardize=True)
    morph_feat_cpu = torch.from_numpy(morph_np).float()  # [N,K]
    sample["morph_feat_cpu"] = morph_feat_cpu
    sample["morph_feat"] = morph_feat_cpu  # alias (will be moved to device for causal)

    sample["morph_feature_names"] = morph_names
    # save feature names for memory/debug
    try:
        out_base0 = ensure_dir(Path(cfg["output"]["out_dir"]) / sample_id)
        (out_base0 / "morph_feature_names.txt").write_text("\n".join(morph_names), encoding="utf-8")
    except Exception:
        pass

    ea3 = sample["edge_attr"]
    ei0 = sample["edge_index"]

    # =========================================================
    # (A) causal: compute I1/I2/I3 ONCE (slow)
    # =========================================================
    if bool(cfg.get("causal", {}).get("enabled", True)):
        ccfg = cfg["causal"]
        print("[stage] causal: init encoder ...")
        t0 = time.time()

        enc = load_prov_gigapath_tile_encoder(
            hf_model_id=ccfg["hf_model_id"],
            timm_prefix=ccfg.get("timm_prefix", "hf-hub:"),
            device=str(device),
            preflight_download=True,
        )
        he_encoder = HEEncoderWrapper(
            encoder=enc,
            input_size=int(ccfg.get("encoder_input_size", 224)),
            finetune=bool(ccfg.get("finetune_encoder", False)),
            normalize_imagenet=bool(ccfg.get("normalize_imagenet", True)),
        )
        builder = CausalGraphBuilder(
            he_encoder=he_encoder,
            enc_out_dim=int(ccfg["enc_out_dim"]),
            extra_he_dim=int(sample.get("morph_feat_cpu", torch.empty((0,0))).shape[1]) if isinstance(sample.get("morph_feat_cpu", None), torch.Tensor) else 0,
            device=str(device),
            proj_device=proj_device,
            z_proj_dim=int(ccfg.get("z_proj_dim", 16)),
            h_proj_dim=int(ccfg.get("h_proj_dim", 16)),
            shrinkage=float(ccfg.get("shrinkage", 1e-3)),
            local_global_ridge=float(ccfg.get("local_global_ridge", 1e-3)),
            min_edges_per_lr=int(ccfg.get("min_edges_per_lr", 20)),
            cmi_clip_percentiles=ccfg.get("cmi_clip_percentiles", (1.0, 99.0)),
            return_debug=bool(ccfg.get("return_debug", False)),
        )
        print(f"[stage] causal: encoder ready. time={time.time()-t0:.1f}s  {gpu_mem_str()}")

        sample_for_causal = copy.copy(sample)
        for k in ["coords", "expr_log", "edge_index", "edge_attr", "morph_feat"]:
            sample_for_causal[k] = sample_for_causal[k].to(device, non_blocking=True)

        print("[stage] causal: build causal graph (this can be slow) .")
        t1 = time.time()

        # Optional: use a pre-built global HE mosaic if provided.
        # Assumption: the current tile (sample["he"]) is centered in that mosaic.
        global_he_path = str(ccfg.get("global_he_path", "")).strip()
        if global_he_path != "":
            fmt_vars = {
                "root": str(dcfg.get("root", "")),
                "dataset_id": str(dcfg.get("dataset_id", "")),
                "sample_id": str(dcfg.get("sample_id", "")),
                "sample_dir": str(sample_for_causal.get("paths", {}).get("sample_dir", "")),
                "he_path": str(sample_for_causal.get("paths", {}).get("he_path", "")),
            }
            try:
                global_he_path = global_he_path.format(**fmt_vars)
            except Exception:
                # If formatting fails, fall back to raw string
                pass
        out = builder(
            sample_for_causal,
            cluster_target=int(ccfg.get("cluster_target", 15)),
            cluster_max=int(ccfg.get("cluster_max", 25)),
            cluster_knn=int(ccfg.get("cluster_knn", 20)),
            cluster_max_edge_dist_px=float(ccfg.get("cluster_max_edge_dist_px", 120.0)),
            mpp=float(dcfg.get("mpp", 0.5)),
            local_um=float(ccfg.get("local_um", 250.0)),
            global_um=float(ccfg.get("global_um", 1000.0)),
            node_patch_um=float(ccfg.get("node_patch_um", ccfg.get("node_um", 80.0))),
            tile_size_px=int(ccfg.get("tile_size_px", 512)),
            global_center_fallback=str(ccfg.get("global_center_fallback", "repeat")),
            global_he_path=global_he_path,
            patch_batch_size=int(ccfg.get("patch_batch_size", 64)),
            amp=True,
        )
        ei, ea3 = graph_as_tensors(out["graph"])
        I1 = out["I1"]
        I2 = out["I2"]
        I3 = out["I3"]
        I4 = out["I4"]
        ei = ei.detach().cpu()
        ea3 = ea3.detach().cpu()
        I1 = I1.detach().cpu()
        I2 = I2.detach().cpu()
        I3 = I3.detach().cpu()
        I4 = I4.detach().cpu()
        print(f"[stage] causal: done. time={time.time()-t1:.1f}s  {gpu_mem_str()}")

        for k in ["coords", "expr_log", "edge_index", "edge_attr", "morph_feat"]:
            if k in sample_for_causal:
                del sample_for_causal[k]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        del builder, he_encoder, enc
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        ei = ei0.detach().cpu()
        ea3 = ea3.detach().cpu()
        E = int(ei.size(1))
        I1 = torch.zeros((E,), dtype=torch.float32)
        I2 = torch.zeros((E,), dtype=torch.float32)
        I3 = torch.zeros((E,), dtype=torch.float32)
        I4 = torch.zeros((E,), dtype=torch.float32)

    # =========================================================
    # (B) node inputs (for node embedding)
    # =========================================================
    node_cfg = cfg.get("node_embedding", {})
    use_node_embed = bool(node_cfg.get("enabled", False))

    expr_log_cpu = sample["expr_log"].detach().cpu().float()
    coords_cpu = sample["coords"].detach().cpu().float()

    he_emb_cpu = None
    he_patches_u8 = None

    if use_node_embed:
        he_np = sample["he"]
        mask_np = sample["mask"]
        coords_xy = coords_cpu.numpy()
        cell_ids = None
        if "cell_ids" in sample and sample["cell_ids"] is not None:
            try:
                cell_ids = sample["cell_ids"].detach().cpu().numpy()
            except Exception:
                cell_ids = np.asarray(sample["cell_ids"])

        # NEW default: node-centric square patch ~80um.
        # With mpp=0.5um/px, 80um -> 160px.
        mpp_val = float(dcfg.get("mpp", 0.5))
        if "he_patch_size_px" in node_cfg and node_cfg.get("he_patch_size_px", None) is not None:
            ps = int(node_cfg.get("he_patch_size_px"))
        else:
            patch_um = float(node_cfg.get("he_patch_um", 80.0))
            ps = int(round(patch_um / max(mpp_val, 1e-6)))
        ps = max(32, ps)
        if (ps % 2) == 1:
            ps += 1
        he_finetune = bool(node_cfg.get("finetune_he_encoder", False))
        precompute_he = bool(node_cfg.get("precompute_he_embedding_if_frozen", True))

        need_patches = he_finetune or (not precompute_he)
        if need_patches:
            print(f"[stage] node_embed: building per-cell HE patches (ps={ps}) ...")
            t0 = time.time()
            patches_np = build_cell_he_patches(
                he=he_np,
                mask=None,
                coords_xy=coords_xy,
                cell_ids=cell_ids,
                patch_size_px=ps,
                mode="square",
            )
            he_patches_u8 = torch.from_numpy(patches_np)
            print(f"[stage] node_embed: patches ready. time={time.time()-t0:.1f}s")

        if (not he_finetune) and precompute_he:
            print("[stage] node_embed: precompute frozen HE embedding ...")
            t0 = time.time()

            if he_patches_u8 is None:
                patches_np = build_cell_he_patches(
                    he=he_np,
                    mask=None,
                    coords_xy=coords_xy,
                    cell_ids=cell_ids,
                    patch_size_px=ps,
                    mode="square",
                )
                he_patches_u8 = torch.from_numpy(patches_np)

            tmp = NodeFeatureEncoder(
                gene_dim=int(expr_log_cpu.size(1)),
                st_hidden=int(node_cfg.get("st_hidden", 256)),
                st_out_dim=int(node_cfg.get("st_out_dim", 128)),
                st_num_layers=int(node_cfg.get("st_num_layers", 2)),
                st_dropout=float(node_cfg.get("st_dropout", 0.1)),
                he_hf_model_id=str(node_cfg.get("hf_model_id")),
                he_timm_prefix=str(node_cfg.get("timm_prefix", "hf-hub:")),
                he_input_size=int(node_cfg.get("encoder_input_size", 224)),
                he_out_dim=int(node_cfg.get("enc_out_dim")),
                he_patch_size_px=cfg_get_int(node_cfg, "he_patch_size_px", ps),
                he_batch_size=int(node_cfg.get("he_batch_size", 64)),
                he_finetune=False,
                he_normalize_imagenet=bool(node_cfg.get("normalize_imagenet", True)),
                precompute_he_if_frozen=True,
                device=device,
            ).to(device)

            he_emb_cpu = tmp.precompute_he_embedding(he_patches_u8)
            del tmp
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if not he_finetune:
                he_patches_u8 = None

            print(f"[stage] node_embed: HE embedding precomputed. time={time.time()-t0:.1f}s  {gpu_mem_str()}")

    lr_id_all = ea3[:, 2].long().numpy()
    lr_vocab = int(lr_id_all.max()) + 1 if lr_id_all.size else 1

    node_pack = {
        "use_node_embed": use_node_embed,
        "expr_log_cpu": expr_log_cpu,
        "coords_cpu": coords_cpu,
        "he_patches_u8": he_patches_u8,
        "he_emb_cpu": he_emb_cpu,
        "morph_feat_cpu": morph_feat_cpu,
        "morph_dim": int(morph_feat_cpu.size(1)),
        "gene_dim": int(expr_log_cpu.size(1)),
        "N": int(expr_log_cpu.size(0)),
        # --- needed for periodic CMI recomputation (sync HE encoder weights) ---
        "sample": sample,
        "sample_id": sample_id,
        "dataset_cfg": dcfg,
    }

    return sample, sample_id, node_pack, ei, ea3, I1, I2, I3, I4, lr_vocab


# =========================================================
# Train + infer one run
# =========================================================
def train_and_infer_one_run(
    cfg: Dict[str, Any],
    run_seed: int,
    node_pack: Dict[str, Any],
    ei_cpu: torch.Tensor,
    ea3_cpu: torch.Tensor,
    I1_cpu: torch.Tensor,
    I2_cpu: torch.Tensor,
    I3_cpu: torch.Tensor,
    I4_cpu: torch.Tensor,
    lr_vocab: int,
    device: torch.device,
):
    set_seed(run_seed)

    mcfg = cfg["model"]
    tcfg = cfg["train"]
    scfg = cfg.get("ssl", {})

    # -------------------------------------------------
    # Needed by periodic CMI recomputation
    # -------------------------------------------------
    proj_device = str(cfg.get("proj_device", "cpu"))
    sample: Dict[str, Any] = node_pack.get("sample", None)
    dcfg: Dict[str, Any] = node_pack.get("dataset_cfg", cfg.get("dataset", {}))
    if sample is None:
        raise RuntimeError(
            "node_pack['sample'] is missing. Ensure build_graph_and_features() attaches the sample dict to node_pack "
            "so train-time CMI recomputation can access it."
        )

    # =========================================================
    # Node feature encoder (HE+ST) - OPTIONAL
    # =========================================================
    node_cfg = cfg.get("node_embedding", {})
    use_node_embed = bool(node_pack.get("use_node_embed", False))

    expr_log_cpu: torch.Tensor = node_pack["expr_log_cpu"]
    coords_cpu: torch.Tensor = node_pack["coords_cpu"]
    he_patches_u8: Optional[torch.Tensor] = node_pack.get("he_patches_u8", None)
    he_emb_cpu: Optional[torch.Tensor] = node_pack.get("he_emb_cpu", None)

    add_coords = bool(mcfg.get("add_coords", False))
    st_batch_size = int(node_cfg.get("st_batch_size", 256))

    node_encoder: Optional[NodeFeatureEncoder] = None

    if use_node_embed:
        node_encoder = NodeFeatureEncoder(
            gene_dim=int(node_pack["gene_dim"]),
            st_hidden=int(node_cfg.get("st_hidden", 256)),
            st_out_dim=int(node_cfg.get("st_out_dim", 128)),
            st_num_layers=int(node_cfg.get("st_num_layers", 2)),
            st_dropout=float(node_cfg.get("st_dropout", 0.1)),
            he_hf_model_id=str(node_cfg["hf_model_id"]),
            he_timm_prefix=str(node_cfg.get("timm_prefix", "hf-hub:")),
            he_input_size=int(node_cfg.get("encoder_input_size", 224)),
            he_out_dim=int(node_cfg["enc_out_dim"]),
            he_patch_size_px=cfg_get_int(node_cfg, "he_patch_size_px", 100),
            he_batch_size=int(node_cfg.get("he_batch_size", 64)),
            he_finetune=bool(node_cfg.get("finetune_he_encoder", False)),
            he_normalize_imagenet=bool(node_cfg.get("normalize_imagenet", True)),
            precompute_he_if_frozen=bool(node_cfg.get("precompute_he_embedding_if_frozen", True)),
            device=device,
        ).to(device)

        if bool(node_cfg.get("finetune_he_encoder", False)) and he_patches_u8 is None:
            raise RuntimeError("node_embedding.finetune_he_encoder=True requires he_patches_u8 to be built in build_graph_and_features().")

        node_in_dim = int(node_cfg["enc_out_dim"]) + int(node_pack.get("morph_dim", 0)) + int(node_cfg.get("st_out_dim", 128))
        if add_coords:
            node_in_dim += 2
    else:
        x_cpu = expr_log_cpu
        if add_coords:
            x_cpu = torch.cat([x_cpu, coords_cpu], dim=1)
        node_in_dim = int(x_cpu.size(1))

    # =========================================================
    # Graph encoder/decoder
    # =========================================================
    encoder = EdgeAwareEncoder(
        in_dim=node_in_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        out_dim=int(mcfg["out_dim"]),
        num_layers=int(mcfg["num_layers"]),
        heads=int(mcfg["num_heads"]),
        lr_vocab=lr_vocab,
        lr_embed_dim=int(mcfg.get("lr_embed_dim", 32)),
        edge_mlp_hidden=int(mcfg.get("edge_mlp_hidden", 64)),
        edge_bias_dim=int(mcfg.get("edge_bias_dim", 32)),
        dropout=float(mcfg.get("dropout", 0.2)),
    ).to(device)

    decoder = EdgeDecoder(
        node_dim=int(mcfg["out_dim"]),
        lr_vocab=lr_vocab,
        lr_embed_dim=int(scfg.get("dec_lr_embed_dim", mcfg.get("lr_embed_dim", 32))),
        hidden=int(scfg.get("dec_hidden", 128)),
        dropout=float(scfg.get("dec_dropout", mcfg.get("dropout", 0.2))),
        use_coexpr=bool(scfg.get("dec_use_coexpr", False)),
        use_causal=bool(scfg.get("dec_use_causal", False)),
    ).to(device)

    # =========================================================
    # Optimizer (include node_encoder modules)
    # =========================================================
    params = list(encoder.parameters()) + list(decoder.parameters())
    if node_encoder is not None:
        params += list(node_encoder.st_mlp.parameters())
        if bool(node_cfg.get("finetune_he_encoder", False)):
            params += list(node_encoder.he_encoder.parameters())

    opt = torch.optim.AdamW(
        params,
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )

    dist_sigma = float(mcfg.get("dist_sigma", 120.0))
    coexpr_log1p = bool(mcfg.get("coexpr_log1p", True))
    coexpr_scale = float(mcfg.get("coexpr_scale", 10.0))

    temperature = float(mcfg.get("temperature", mcfg.get("tempture", 1.0)))
    temperature = max(temperature, 1e-6)

    ei_full = ei_cpu.to(device, non_blocking=True)
    ea3_full = ea3_cpu.to(device, non_blocking=True)
    I1_full = I1_cpu.to(device, non_blocking=True).float()
    I2_full = I2_cpu.to(device, non_blocking=True).float()
    I3_full = I3_cpu.to(device, non_blocking=True).float()
    I4_full = I4_cpu.to(device, non_blocking=True).float()

    E_full = int(ei_full.size(1))

    # =========================================================
    # SSL config (masked-edge + rank + consistency)
    # =========================================================
    epochs = int(tcfg.get("epochs", 200))
    print_every = int(tcfg.get("print_every", 25))
    grad_clip = float(tcfg.get("grad_clip", 5.0))

    num_mask_edges = int(scfg.get("num_mask_edges", max(512, min(2048, E_full // 20))))
    beta_rank = float(scfg.get("beta_rank", 0.0))
    gamma_cons = float(scfg.get("gamma_cons", 0.0))
    num_rank_pairs = int(scfg.get("num_rank_pairs", 1024))
    mask_weight = float(scfg.get("mask_weight", 1.0))

    # teacher weights (edge prior) for rank sampling
    with torch.no_grad():
        raw0 = I1_full + 1.0 * I2_full + (-1.0) * I3_full
        mu0 = raw0.mean()
        std0 = raw0.std(unbiased=False)
        teacher_w0 = torch.sigmoid((raw0 - mu0) / (std0 + 1e-6))

    def _recompute_cmi_with_synced_encoder() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute causal scores I1..I4 (and possibly updated graph) using the CURRENT node HE encoder weights.

        Returns:
            (ei_cpu_new, ea3_cpu_new, I1_cpu_new, I2_cpu_new, I3_cpu_new, I4_cpu_new)
        """
        if node_encoder is None:
            raise RuntimeError("node_encoder is None: cannot sync encoder weights for CMI recomputation.")
        if not getattr(node_encoder, "he_finetune", False):
            # Even if not finetuning, allow recompute for consistency (no weight change).
            pass

        ccfg = cfg["causal"]
        dcfg = cfg["dataset"]

        # (1) Init a fresh causal encoder and SYNC weights from node encoder
        enc_new = load_prov_gigapath_tile_encoder(
            hf_model_id=ccfg["hf_model_id"],
            timm_prefix=ccfg.get("timm_prefix", "hf-hub:"),
            device=str(device),
            preflight_download=False,
        )
        try:
            enc_new.load_state_dict(node_encoder.he_encoder.encoder.state_dict(), strict=True)
        except Exception:
            # allow minor key mismatches (e.g. heads) if any
            enc_new.load_state_dict(node_encoder.he_encoder.encoder.state_dict(), strict=False)

        he_encoder_new = HEEncoderWrapper(
            encoder=enc_new,
            input_size=int(ccfg.get("encoder_input_size", 224)),
            finetune=False,  # CMI path is non-differentiable; keep frozen during recompute
            normalize_imagenet=bool(ccfg.get("normalize_imagenet", True)),
        )

        builder_new = CausalGraphBuilder(
            he_encoder=he_encoder_new,
            enc_out_dim=int(ccfg["enc_out_dim"]),
            extra_he_dim=int(node_pack.get("morph_feat_cpu", torch.empty((0, 0))).shape[1]) if isinstance(node_pack.get("morph_feat_cpu", None), torch.Tensor) else 0,
            device=str(device),
            proj_device=proj_device,
            z_proj_dim=int(ccfg.get("z_proj_dim", 16)),
            h_proj_dim=int(ccfg.get("h_proj_dim", 16)),
            shrinkage=float(ccfg.get("shrinkage", 1e-3)),
            local_global_ridge=float(ccfg.get("local_global_ridge", 1e-3)),
            min_edges_per_lr=int(ccfg.get("min_edges_per_lr", 20)),
            cmi_clip_percentiles=ccfg.get("cmi_clip_percentiles", (1.0, 99.0)),
            return_debug=bool(ccfg.get("return_debug", False)),
        )

        # (2) Prepare sample for causal (move required tensors to device)
        sample_for_causal = copy.copy(sample)
        sample_for_causal["edge_index"] = ei_cpu.to(device, non_blocking=True)
        sample_for_causal["edge_attr"] = ea3_cpu.to(device, non_blocking=True)
        sample_for_causal["coords"] = coords_cpu.to(device, non_blocking=True)
        sample_for_causal["expr_log"] = expr_log_cpu.to(device, non_blocking=True)
        if "morph_feat_cpu" in node_pack and isinstance(node_pack["morph_feat_cpu"], torch.Tensor):
            sample_for_causal["morph_feat"] = node_pack["morph_feat_cpu"].to(device, non_blocking=True)

        # Optional global HE mosaic if configured
        global_he_path = str(ccfg.get("global_he_path", "")).strip()
        if global_he_path != "":
            fmt_vars = {"root": str(dcfg.get("root", ""))}
            try:
                global_he_path = global_he_path.format(**fmt_vars)
            except Exception:
                pass

        # (3) Recompute
        out_new = builder_new(
            sample_for_causal,
            cluster_target=int(ccfg.get("cluster_target", 15)),
            cluster_max=int(ccfg.get("cluster_max", 25)),
            cluster_knn=int(ccfg.get("cluster_knn", 20)),
            cluster_max_edge_dist_px=float(ccfg.get("cluster_max_edge_dist_px", 120.0)),
            mpp=float(dcfg.get("mpp", 0.5)),
            local_um=float(ccfg.get("local_um", 250.0)),
            global_um=float(ccfg.get("global_um", 1000.0)),
            node_patch_um=float(ccfg.get("node_patch_um", ccfg.get("node_um", 80.0))),
            tile_size_px=int(ccfg.get("tile_size_px", 512)),
            global_center_fallback=str(ccfg.get("global_center_fallback", "repeat")),
            global_he_path=global_he_path,
            patch_batch_size=int(ccfg.get("patch_batch_size", 64)),
            amp=True,
        )

        ei_new, ea3_new = graph_as_tensors(out_new["graph"])
        I1_new = out_new["I1"]
        I2_new = out_new["I2"]
        I3_new = out_new["I3"]
        I4_new = out_new["I4"]

        # (4) Detach to CPU
        ei_cpu_new = ei_new.detach().cpu()
        ea3_cpu_new = ea3_new.detach().cpu()
        I1_cpu_new = I1_new.detach().cpu()
        I2_cpu_new = I2_new.detach().cpu()
        I3_cpu_new = I3_new.detach().cpu()
        I4_cpu_new = I4_new.detach().cpu()

        # cleanup
        for k in ["coords", "expr_log", "edge_index", "edge_attr", "morph_feat"]:
            if k in sample_for_causal:
                del sample_for_causal[k]
        del builder_new, he_encoder_new, enc_new
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return ei_cpu_new, ea3_cpu_new, I1_cpu_new, I2_cpu_new, I3_cpu_new, I4_cpu_new


    def compute_node_x() -> torch.Tensor:
        if node_encoder is None:
            x0 = expr_log_cpu.to(device, non_blocking=True)
            if add_coords:
                x0 = torch.cat([x0, coords_cpu.to(device, non_blocking=True)], dim=1)
            return x0
        return node_encoder(
            expr_log_cpu=expr_log_cpu,
            coords_cpu=coords_cpu,
            add_coords=add_coords,
            he_patches_u8=he_patches_u8,
            he_emb_cpu=he_emb_cpu,
            morph_feat_cpu=node_pack.get("morph_feat_cpu", None),
            st_batch_size=st_batch_size,
        )

    # =========================================================
    # Train
    # =========================================================
    encoder.train()
    decoder.train()
    if node_encoder is not None:
        node_encoder.train()

    for ep in range(1, epochs + 1):
        # ==========================
        # _CMI_RECOMPUTE_EVERY_10_EPOCHS
        # Recompute CMI scores every 10 epochs using the latest node HE encoder weights.
        # This is an outer-loop update (non-differentiable), similar to EM / alternating optimization.
        # ==========================
        if bool(cfg.get("causal", {}).get("enabled", True)) and (ep % 10 == 0):
            if node_encoder is not None and getattr(node_encoder, "he_finetune", False):
                print(f"[stage] causal: recompute CMI @ epoch={ep} (sync HE encoder weights) ...")
                t_cmi = time.time()
                ei_cpu_new, ea3_cpu_new, I1_cpu_new, I2_cpu_new, I3_cpu_new, I4_cpu_new = _recompute_cmi_with_synced_encoder()
                # swap in new tensors (CPU + GPU)
                ei_cpu = ei_cpu_new
                ea3_cpu = ea3_cpu_new
                I1_cpu = I1_cpu_new
                I2_cpu = I2_cpu_new
                I3_cpu = I3_cpu_new
                I4_cpu = I4_cpu_new
                ei_full = ei_cpu.to(device, non_blocking=True)
                ea3_full = ea3_cpu.to(device, non_blocking=True)
                I1_full = I1_cpu.to(device, non_blocking=True).float()
                I2_full = I2_cpu.to(device, non_blocking=True).float()
                I3_full = I3_cpu.to(device, non_blocking=True).float()
                I4_full = I4_cpu.to(device, non_blocking=True).float()
                E_full = int(ei_full.size(1))
                # refresh teacher prior for rank sampling
                with torch.no_grad():
                    raw0 = I1_full
                    mu0 = raw0.mean()
                    std0 = raw0.std(unbiased=False)
                    teacher_w0 = torch.sigmoid((raw0 - mu0) / (std0 + 1e-6))
                print(f"[stage] causal: recompute done. time={time.time()-t_cmi:.1f}s  E={E_full}  {gpu_mem_str()}")
            else:
                # if node encoder isn't finetuned, recomputation won't reflect weight updates; skip to save time
                pass

        opt.zero_grad(set_to_none=True)

        x_ep = compute_node_x()

        mask_e = torch.randint(0, E_full, (num_mask_edges,), device=device)
        keep_mask = torch.ones((E_full,), dtype=torch.bool, device=device)
        keep_mask[mask_e] = False
        ei_m = ei_full[:, keep_mask]
        ea_m3 = ea3_full[keep_mask]
        I1_m = I1_full[keep_mask]
        I2_m = I2_full[keep_mask]
        I3_m = I3_full[keep_mask]
        I4_m = I4_full[keep_mask]

        ea_m = encoder.compose_edge_attr7(ea_m3, I1_m, I2_m, I3_m, I4_m)

        z, _ = encoder(x_ep, ei_m, ea_m, dist_sigma, coexpr_log1p, coexpr_scale, temperature, True)
        pred_logit_m = decoder(z, ei_m, ea_m)
        pred_m = torch.sigmoid(pred_logit_m)

        target = teacher_w0[keep_mask]
        L_mask = F.mse_loss(pred_m, target) * mask_weight

        # rank loss
        L_rank = torch.tensor(0.0, device=device)
        if beta_rank > 0 and num_rank_pairs > 0:
            pos_idx, neg_idx = sample_rank_pairs_same_src(ei_full, teacher_w0, num_rank_pairs, device=device)
            if pos_idx is not None and neg_idx is not None and pos_idx.numel() > 0:
                ea7_full = encoder.compose_edge_attr7(ea3_full, I1_full, I2_full, I3_full, I4_full)
                z_full, _ = encoder(x_ep, ei_full, ea7_full, dist_sigma, coexpr_log1p, coexpr_scale, temperature, True)
                pred_pos = decoder(z_full, ei_full[:, pos_idx], ea7_full[pos_idx])
                pred_neg = decoder(z_full, ei_full[:, neg_idx], ea7_full[neg_idx])
                L_rank = bpr_ranking_loss(pred_pos, pred_neg)

        # consistency loss
        L_cons = torch.tensor(0.0, device=device)
        if gamma_cons > 0:
            mask_e2 = torch.randint(0, E_full, (num_mask_edges,), device=device)
            keep_mask2 = torch.ones((E_full,), dtype=torch.bool, device=device)
            keep_mask2[mask_e2] = False
            ei_m2 = ei_full[:, keep_mask2]
            ea_m2 = encoder.compose_edge_attr7(ea3_full[keep_mask2], I1_full[keep_mask2], I2_full[keep_mask2], I3_full[keep_mask2], I4_full[keep_mask2])
            z2, _ = encoder(x_ep, ei_m2, ea_m2, dist_sigma, coexpr_log1p, coexpr_scale, temperature, True)
            pred_logit_m2 = decoder(z2, ei_m2, ea_m2)
            pred_m2 = torch.sigmoid(pred_logit_m2)
            L_cons = (pred_m.mean() - pred_m2.mean()).pow(2)

        loss = L_mask + beta_rank * L_rank + gamma_cons * L_cons
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        opt.step()

        if ep == 1 or ep == epochs or (ep % print_every) == 0:
            print(
                f"  [run seed={run_seed}] ep={ep:4d}/{epochs} "
                f"loss={loss.item():.6f}"
                f"tau={temperature:g} {gpu_mem_str()}"
            )

    # =========================================================
    # Infer strength with final (UPDATED) params
    # strength = pre-softmax attn logit (e_ij), averaged over heads
    # =========================================================
    encoder.eval()
    decoder.eval()
    if node_encoder is not None:
        node_encoder.eval()

    with torch.no_grad():
        x_eval = compute_node_x()
        ea7_full = encoder.compose_edge_attr7(ea3_full, I1_full, I2_full, I3_full, I4_full)
        z_full, e_ij = encoder(x_eval, ei_full, ea7_full, dist_sigma, coexpr_log1p, coexpr_scale, temperature, True)
        strength = e_ij.mean(dim=1)  # [E] pre-softmax (you requested: unnormalized = strength)

        # decoder logit is kept only for debugging (not used as final strength/score)
        dec_logit = decoder(z_full, ei_full, ea7_full)

    return (
        {"encoder": encoder, "decoder": decoder, "node_encoder": node_encoder},
        strength.detach().cpu().numpy().astype(np.float32),
        dec_logit.detach().cpu().numpy().astype(np.float32),
    )


# =========================================================
# Export (strength + score)
# =========================================================
def export_results(
    cfg: Dict[str, Any],
    dataset_id: str,
    sample: Dict[str, Any],
    sample_id: str,
    ei_cpu: torch.Tensor,
    ea7_cpu: torch.Tensor,
    strength: np.ndarray,   # pre-softmax, unnormalized
    score: np.ndarray,      # minmax_01(strength)
    keep_mask: np.ndarray,  # boolean mask over edges (E,)
    out_root: Path,
    tag: str = "",          # "" (filtered) or "_all" (unfiltered)
):
    """
    Save only two CSVs:
      - edges_with_scores{tag}.csv
      - pred_edges{tag}.csv

    tag=""     -> filtered (use keep_mask passed from caller)
    tag="_all" -> unfiltered full export (caller should pass keep_mask=np.ones(E))
    """
    src = ei_cpu[0].numpy().astype(np.int64)
    dst = ei_cpu[1].numpy().astype(np.int64)
    ea = ea7_cpu.numpy().astype(np.float32)

    dist = ea[:, 0]
    coexpr = ea[:, 1]
    lr_id = ea[:, 2].astype(np.int64)
    cmi1 = ea[:, 3]
    cmi2 = ea[:, 4]
    cmi3 = ea[:, 5]
    cmi4 = ea[:, 6]

    lr_meta = build_lr_meta_map(sample["lr_pairs"]["pairs_kept"])
    cell_ids = sample.get("cell_ids", None)
    if cell_ids is None:
        cell_ids = [str(i) for i in range(int(sample["coords"].shape[0]))]

    keep_mask = np.asarray(keep_mask, dtype=bool)
    idx_keep = np.where(keep_mask)[0]

    rows = []
    for e in idx_keep.tolist():
        pid = int(lr_id[e])
        meta = lr_meta.get(pid, {"ligand": "", "receptor": "", "pathway": "", "mechanism": ""})
        rows.append({
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "src_idx": int(src[e]),
            "dst_idx": int(dst[e]),
            "src_cell_id": str(cell_ids[int(src[e])]),
            "dst_cell_id": str(cell_ids[int(dst[e])]),
            "ligand": meta["ligand"],
            "receptor": meta["receptor"],
            "pathway": meta["pathway"],
            "mechanism": meta["mechanism"],
            "distance_px": float(dist[e]),
            "coexpr": float(coexpr[e]),
            "cmi1": float(cmi1[e]),
            "cmi2": float(cmi2[e]),
            "cmi3": float(cmi3[e]),
            "cmi4": float(cmi4[e]),
            "strength": float(strength[e]),  # unnormalized
            "score": float(score[e]),        # normalized [0,1]
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["score"], ascending=[False]).reset_index(drop=True)

    edges_path = out_root / f"edges_with_scores{tag}.csv"
    pred_path = out_root / f"pred_edges{tag}.csv"

    # full table
    df.to_csv(edges_path, index=False)

    # compact table (only score)
    compact_cols_score = [
        "dataset_id", "sample_id",
        "src_cell_id", "dst_cell_id",
        "ligand", "receptor",
        "pathway", "mechanism",
        "score",
    ]
    if len(df) > 0:
        df[compact_cols_score].to_csv(pred_path, index=False)
    else:
        pd.DataFrame(columns=compact_cols_score).to_csv(pred_path, index=False)

    print(f"[export{tag}] kept_edges={len(df)}  files saved: {edges_path.name}, {pred_path.name}")


# =========================================================
# Main
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_ccc.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg.get("seed", 7))
    R = int(cfg.get("R", 1))
    device = torch.device(cfg.get("device", "cuda"))
    proj_device = str(cfg.get("proj_device", "cpu"))

    dataset_id = infer_dataset_id_from_root(cfg["dataset"]["root"])

    print("[stage] dataset+graph: loading ...")
    sample, sample_id, node_pack, ei_cpu, ea3_cpu, I1_cpu, I2_cpu, I3_cpu, I4_cpu, lr_vocab = build_graph_and_features(cfg, device, proj_device)

    N = int(node_pack["N"])
    E = int(ea3_cpu.size(0))

    print(f"\n[info] dataset_id={dataset_id}  sample_id={sample_id}  N={N}  E={E}  R={R}")
    print(f"[info] lr_vocab={lr_vocab}")
    print(f"[info] {gpu_mem_str()}")

    ei_np = ei_cpu.numpy().astype(np.int64)
    deg = degree_stats(ei_np, N)
    ea_np = ea3_cpu.numpy().astype(np.float32)
    dist = ea_np[:, 0]
    coexpr = ea_np[:, 1]
    lr_id = ea_np[:, 2].astype(np.int64)

    I1_np = I1_cpu.numpy().astype(np.float32)
    I2_np = I2_cpu.numpy().astype(np.float32)
    I3_np = I3_cpu.numpy().astype(np.float32)
    I4_np = I4_cpu.numpy().astype(np.float32)

    uniq_lr = int(np.unique(lr_id).size) if lr_id.size else 0
    edges_per_lr = np.bincount(lr_id) if lr_id.size else np.array([], dtype=np.int64)

    out_base = ensure_dir(Path(cfg["output"]["out_dir"]) / sample_id)
    (out_base / "runs").mkdir(parents=True, exist_ok=True)

    # runs storage
    strength_runs: List[np.ndarray] = []
    dec_logit_runs: List[np.ndarray] = []
    alpha_runs: List[float] = []
    beta_runs: List[float] = []
    gamma_runs: List[float] = []
    delta_runs: List[float] = []

    run_iter = range(R)
    if tqdm is not None and R > 1:
        run_iter = tqdm(run_iter, desc="runs", ncols=110)

    for r in run_iter:
        run_seed = base_seed + 1000 * r
        print(f"\n==== Run {r+1}/{R} (seed={run_seed}) ====")
        models, strength_r, dec_logit_r = train_and_infer_one_run(
            cfg, run_seed, node_pack, ei_cpu, ea3_cpu, I1_cpu, I2_cpu, I3_cpu, I4_cpu, lr_vocab, device
        )

        alpha_runs.append(float(models["encoder"].alpha.detach().cpu().item()))
        beta_runs.append(float(models["encoder"].beta.detach().cpu().item()))
        gamma_runs.append(float(models["encoder"].gamma.detach().cpu().item()))
        delta_runs.append(float(models["encoder"].delta.detach().cpu().item()))

        strength_runs.append(strength_r)
        dec_logit_runs.append(dec_logit_r)

        run_dir = out_base / "runs" / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # per-run outputs
        np.save(run_dir / "strength.npy", strength_r.astype(np.float32))
        np.save(run_dir / "score.npy", minmax_01(strength_r).astype(np.float32))
        np.save(run_dir / "decoder_logit.npy", dec_logit_r.astype(np.float32))

        if bool(cfg["output"].get("save_model", True)):
            torch.save(
                {
                    "encoder": models["encoder"].state_dict(),
                    "decoder": models["decoder"].state_dict(),
                    "node_encoder": (models["node_encoder"].state_dict() if models.get("node_encoder", None) is not None else None),
                    "cfg": cfg,
                    "seed": run_seed,
                },
                run_dir / "model.pt",
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    strength_runs_np = np.stack(strength_runs, axis=0)  # [R,E]
    dec_logit_runs_np = np.stack(dec_logit_runs, axis=0)  # [R,E]

    ens = cfg.get("ensemble", {})
    agg = ens.get("agg", "mean")

    # =========================
    # Final outputs:
    # strength = unnormalized pre-softmax logits
    # score    = normalized strength in [0,1]
    # =========================
    strength_final = aggregate_strength(strength_runs_np, agg).astype(np.float32)
    score_final = minmax_01(strength_final).astype(np.float32)

    ocfg = cfg["output"]
    use_min_keep = bool(ocfg.get("use_min_keep", False))

    # per-run keep (use score per run for filtering)
    per_run_keep = []
    for r in range(R):
        per_run_keep.append(
            global_topk_mask(
                strength=minmax_01(strength_runs_np[r]),
                keep_top_percent=float(ocfg.get("keep_top_percent", 1.0)),
                min_keep=int(ocfg.get("min_keep", 10)),
                max_keep=int(ocfg.get("max_keep", 2000)),
                use_min_keep=use_min_keep,
            )
        )
    per_run_keep = np.stack(per_run_keep, axis=0)
    presence = per_run_keep.mean(axis=0).astype(np.float32)

    if bool(ens.get("use_stability", False)) and R > 1:
        min_frac = float(ens.get("min_presence_frac", 0.6))
        keep_mask = presence >= min_frac
        print(f"[filter] stability ON: kept={keep_mask.sum()}/{len(keep_mask)} (min_presence_frac={min_frac})")
    else:
        keep_mask = global_topk_mask(
            strength=score_final,  # use normalized score for final filtering
            keep_top_percent=float(ocfg.get("keep_top_percent", 1.0)),
            min_keep=int(ocfg.get("min_keep", 10)),
            max_keep=int(ocfg.get("max_keep", 2000)),
            use_min_keep=use_min_keep,
        )
        print(f"[filter] stability OFF: kept={keep_mask.sum()}/{len(keep_mask)}  (use_min_keep={use_min_keep})")

    with open(out_base / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # save main npy
    np.save(out_base / "strength.npy", strength_final.astype(np.float32))
    np.save(out_base / "score.npy", score_final.astype(np.float32))
    np.save(out_base / "presence_frac.npy", presence.astype(np.float32))
    np.save(out_base / "decoder_logit_ensemble.npy", aggregate_strength(dec_logit_runs_np, agg).astype(np.float32))

    # Use mean α/β/γ/δ to export the 4 scaled CMI features (each mapped to ~[0,1])
    alpha_mean = float(np.mean(alpha_runs)) if len(alpha_runs) > 0 else 1.0
    beta_mean  = float(np.mean(beta_runs))  if len(beta_runs)  > 0 else 1.0
    gamma_mean = float(np.mean(gamma_runs)) if len(gamma_runs) > 0 else 1.0
    delta_mean = float(np.mean(delta_runs)) if len(delta_runs) > 0 else 1.0

    def _norm01_t(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.float()
        mu = x.mean()
        std = x.std(unbiased=False)
        return torch.sigmoid((x - mu) / (std + eps))

    cmi1_01 = _norm01_t(I1_cpu.float() * alpha_mean)
    cmi2_01 = _norm01_t(I2_cpu.float() * beta_mean)
    cmi3_01 = _norm01_t(I3_cpu.float() * gamma_mean)
    cmi4_01 = _norm01_t(I4_cpu.float() * delta_mean)

    ea7_cpu_export = torch.cat(
        [ea3_cpu.float(),
         cmi1_01.view(-1, 1),
         cmi2_01.view(-1, 1),
         cmi3_01.view(-1, 1),
         cmi4_01.view(-1, 1)],
        dim=1
    )

    cmi1_np = cmi1_01.detach().cpu().numpy().astype(np.float32)
    cmi2_np = cmi2_01.detach().cpu().numpy().astype(np.float32)
    cmi3_np = cmi3_01.detach().cpu().numpy().astype(np.float32)
    cmi4_np = cmi4_01.detach().cpu().numpy().astype(np.float32)

    diag = {
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "N": N,
        "E": int(ea_np.shape[0]),
        "degree": deg,
        "edge_stats": {
            "distance_px": np_stats(dist, "distance_px"),
            "coexpr": np_stats(coexpr, "coexpr"),
            "cmi1": np_stats(cmi1_np, "cmi1"),
            "cmi2": np_stats(cmi2_np, "cmi2"),
            "cmi3": np_stats(cmi3_np, "cmi3"),
            "cmi4": np_stats(cmi4_np, "cmi4"),
            "strength": np_stats(strength_final, "strength"),
            "score": np_stats(score_final, "score"),
        },
        "lr_stats": {
            "unique_lr_in_graph": uniq_lr,
            "pairs_kept_total": int(len(sample["lr_pairs"]["pairs_kept"])),
            "edges_per_lr": np_stats(edges_per_lr, "edges_per_lr") if edges_per_lr.size else {"name": "edges_per_lr", "size": 0},
        },
        "learned_cmi_params": {
            "alpha_runs": alpha_runs,
            "beta_runs": beta_runs,
            "gamma_runs": gamma_runs,
            "delta_runs": delta_runs,
            "alpha_mean": alpha_mean,
            "beta_mean": beta_mean,
            "gamma_mean": gamma_mean,
            "delta_mean": delta_mean,
        },
        "config_path": str(args.config),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(out_base / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    # (1) filtered export (uses keep_mask)
    export_results(
        cfg, dataset_id, sample, sample_id,
        ei_cpu, ea7_cpu_export,
        strength_final,
        score_final,
        keep_mask,
        out_base,
        tag="",
    )

    # (2) unfiltered export (full edges)
    keep_mask_all = np.ones((ei_cpu.size(1),), dtype=bool)
    export_results(
        cfg, dataset_id, sample, sample_id,
        ei_cpu, ea7_cpu_export,
        strength_final,
        score_final,
        keep_mask_all,
        out_base,
        tag="_all",
    )

    print("\n[done]")
    print("out_dir:", str(out_base))
    print("saved: diagnostics.json, edges_with_scores.csv, pred_edges.csv, edges_with_scores_all.csv, pred_edges_all.csv")
    print("saved: strength.npy, score.npy, presence_frac.npy, decoder_logit_ensemble.npy")
    print("saved: runs/* (per-run strength/score/decoder_logit and optional models)")


if __name__ == "__main__":
    main()