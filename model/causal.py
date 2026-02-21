# -*- coding: utf-8 -*-
"""
causal.py

核心输出：
- 在原预筛选图基础上，为每条边计算四个因果分量 (I1, I2, I3, I4)
- graph.edge_attr 仍保持 3 维: [distance_px, coexpr, lr_id]
- 额外返回 I1/I2/I3/I4（与边顺序一一对应），用于在训练阶段作为边特征（与基础3维拼接成7维）并由可学习系数缩放
- 返回: {"graph": Graph, "I1": Tensor[E], "I2": Tensor[E], "I3": Tensor[E], "I4": Tensor[E], "debug": ... (可选)}

- H_i：窗口内全部细胞区域（mask>0 的 union）
- DAG 方向：x_{j,k} -> x_{i,k}（sender->receiver）
- CMI：边作为样本；cluster 内离散均匀抽样用“加权二阶矩”解析近似
- encoder 冻结/微调：main 中 FINETUNE_ENCODER 控制
"""

from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.spatial import cKDTree
except Exception as e:
    raise ImportError("scipy is required (scipy.spatial.cKDTree).") from e

try:
    from torch_geometric.data import Data
except Exception:
    Data = None


# ============================================================
# 0) HuggingFace / Prov-GigaPath encoder loading
# ============================================================

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _get_cached_hf_token() -> str:
    """
    Retrieve HF token without asking user to pass it in config.
    Priority:
    1) env vars HF_TOKEN / HUGGINGFACE_HUB_TOKEN
    2) huggingface_hub cached token API
    3) token files:
       - ~/.cache/huggingface/token
       - ~/.huggingface/token
    """
    # 1) env
    t = (os.environ.get("HF_TOKEN", "") or "").strip()
    if t:
        return t
    t = (os.environ.get("HUGGINGFACE_HUB_TOKEN", "") or "").strip()
    if t:
        return t

    # 2) huggingface_hub API
    try:
        from huggingface_hub.utils import get_token  # type: ignore
        t = (get_token() or "").strip()
        if t:
            return t
    except Exception:
        pass

    try:
        from huggingface_hub import HfFolder  # type: ignore
        t = (HfFolder.get_token() or "").strip()
        if t:
            return t
    except Exception:
        pass

    # 3) common token files
    home = os.path.expanduser("~")
    for p in [
        os.path.join(home, ".cache", "huggingface", "token"),
        os.path.join(home, ".huggingface", "token"),
    ]:
        t = _read_text_file(p)
        if t:
            return t

    return ""


def ensure_hf_token_available(set_env_if_missing: bool = True) -> str:
    """
    Ensure a usable HF token exists (from local cache or env).

    IMPORTANT:
    - We DO NOT call huggingface_hub.login() here to avoid the "Note: HF_TOKEN is set ..." message.
    - huggingface_hub will automatically use cached token written by `huggingface-cli login`.
    - Optionally set env vars if missing, for maximum compatibility with timm/hf_hub_download.
    """
    t = _get_cached_hf_token()
    if not t:
        raise RuntimeError(
            "Hugging Face token not found on this server.\n"
            "Fix (one-time): run `huggingface-cli login` (with an account that has access to prov-gigapath).\n"
            "Then rerun causal.py. No token is required in code.\n"
        )

    if set_env_if_missing:
        # Only set if absent, to avoid surprising overrides
        if not (os.environ.get("HF_TOKEN") or "").strip():
            os.environ["HF_TOKEN"] = t
        if not (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip():
            os.environ["HUGGINGFACE_HUB_TOKEN"] = t

    return t


def load_prov_gigapath_tile_encoder(
    hf_model_id: str = "prov-gigapath/prov-gigapath",
    timm_prefix: str = "hf-hub:",
    device: str = "cuda",
    preflight_download: bool = True,
) -> nn.Module:
    """
    Load tile encoder via timm from HF hub.

    - No token required in config/code.
    - Uses cached HF token (huggingface-cli login) or env token.
    - Does NOT call huggingface_hub.login() -> avoids Note message.
    """
    token = ensure_hf_token_available(set_env_if_missing=True)

    if preflight_download:
        # optional: verify access quickly; if it fails, timm will fail too
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            _ = hf_hub_download(repo_id=hf_model_id, filename="model.safetensors", token=token)
        except Exception:
            pass

    try:
        import timm  # type: ignore
    except Exception as e:
        raise ImportError("Please install timm: pip install timm") from e

    model_name = f"{timm_prefix}{hf_model_id}"
    m = timm.create_model(model_name, pretrained=True, num_classes=0)
    m = m.to(torch.device(device))
    m.eval()
    return m


class HEEncoderWrapper(nn.Module):
    """
    输入：RGB patch tensor（[B,H,W,3] 或 [B,3,H,W]，uint8/float）
    输出：embedding [B,D]

    你的最新需求：**只训练最后的 head，其余部分全部冻结**（避免爆显存）。

    设计：
    - backbone(Prov-GigaPath tile encoder / ViT) 永远冻结：`requires_grad=False`
    - forward 时 backbone 永远走 `torch.inference_mode()`（不构建反向图）+ 可选 AMP
    - 在 backbone 输出 embedding 上接一个很小的可学习 head（默认 residual adapter）
      只有这个 head 参与反传与优化。

    这样显存主要由“小 head 的反传”构成，几乎不再受 ViT 反传影响。

    参数：
    - finetune：为了兼容旧接口，这里表示“是否训练 head”
      True  -> head 可学习
      False -> head 冻结（完全推理）
    """
    def __init__(
        self,
        encoder: nn.Module,
        input_size: int = 224,
        finetune: bool = False,
        normalize_imagenet: bool = True,
        head_type: str = "residual_mlp",   # residual_mlp | linear
        head_hidden: int = 0,              # 0 => use D//2
    ):
        super().__init__()
        self.encoder = encoder
        self.input_size = int(input_size)
        self.normalize = bool(normalize_imagenet)

        # backbone 永远冻结（只训 head）
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.head_type = str(head_type)
        self.head_hidden = int(head_hidden)

        # head 延迟构建：第一次 forward 知道 D
        self.head: Optional[nn.Module] = None
        # residual 缩放，初始化为 0：一开始等价于“完全不改 backbone 输出”
        self.head_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

        # 如果能从 timm/encoder 直接拿到 embedding 维度，就在 init 里直接建 head，
        # 避免 optimizer 在第一次 forward 前看不到 head 参数。
        dim = getattr(self.encoder, "num_features", None)
        if dim is None:
            dim = getattr(self.encoder, "embed_dim", None)
        if dim is None:
            dim = getattr(self.encoder, "feature_info", None)
        if isinstance(dim, int) and dim > 0:
            self._maybe_build_head(dim=int(dim), device=next(self.encoder.parameters()).device)

        self.set_finetune(bool(finetune))

    def set_finetune(self, finetune: bool) -> None:
        # 兼容旧接口：finetune==True 表示训练 head；False 表示冻结 head
        self.finetune = bool(finetune)
        self.head_scale.requires_grad_(self.finetune)
        if self.head is not None:
            for p in self.head.parameters():
                p.requires_grad_(getattr(self, 'finetune', False))

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expect 4D tensor, got {tuple(x.shape)}")
        if x.shape[1] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unrecognized image tensor shape={tuple(x.shape)}")

    def _maybe_build_head(self, dim: int, device: torch.device) -> None:
        if self.head is not None:
            return
        d = int(dim)
        if d <= 0:
            raise ValueError(f"Invalid embedding dim: {d}")

        if self.head_type == "linear":
            head = nn.Linear(d, d, bias=True)
            # 初始化为 0：配合 head_scale=0，初始严格 identity
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.head = head
        else:
            # residual MLP: D -> H -> D
            h = self.head_hidden if self.head_hidden > 0 else max(32, d // 2)
            head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, h),
                nn.GELU(),
                nn.Linear(h, d),
            )
            # 最后一层初始化为 0：初始近似 identity（通过 residual）
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.head = head

        self.head.to(device=device)
        for p in self.head.parameters():
            p.requires_grad_(getattr(self, 'finetune', False))

    def forward(self, images: torch.Tensor, amp: bool = True) -> torch.Tensor:
        x = self._to_bchw(images)
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()

        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)

        if self.normalize:
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / (std + 1e-6)

        use_amp = (amp and (x.device.type == "cuda"))

        # backbone：主干永远冻结；如果需要训练 head，用 no_grad（而不是 inference_mode）
        # 这样输出不是 inference tensor，允许被 autograd 保存用于 head 的反传。
        if getattr(self, 'finetune', False):
            ctx = torch.no_grad()
        else:
            ctx = torch.inference_mode()
        with ctx:
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.encoder(x)
            else:
                out = self.encoder(x)

        if isinstance(out, dict):
            for k in ["emb", "embedding", "feat", "features", "pooled", "last_hidden_state"]:
                if k in out:
                    out = out[k]
                    break
        if not torch.is_tensor(out):
            raise TypeError(f"Encoder output must be Tensor or dict-of-Tensor, got {type(out)}")

        if out.ndim == 3:
            out = out.mean(dim=1)
        if out.ndim != 2:
            raise ValueError(f"Encoder output must be [B,D], got {tuple(out.shape)}")

        # build head lazily
        self._maybe_build_head(dim=int(out.shape[1]), device=out.device)

        if not self.finetune:
            # head 冻结：直接返回 backbone embedding
            return out

        # 只在 embedding 上训练一个很小的 head（反传只覆盖 head）
        # 为了数值稳定：head 用 fp32
        base = out.float()
        delta = self.head(base)
        # residual
        out2 = base + self.head_scale * delta
        return out2.to(dtype=out.dtype)

# ============================================================
# 1) Spatial clustering (small clusters)  —— distance + expression-aware, auto K
# ============================================================

def _standardize_features(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise standardization: (x-mean)/std. Returns standardized, mean, std."""
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (x - mu) / sd, mu.reshape(-1), sd.reshape(-1)


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ init. X: [N,D] -> centers: [K,D]."""
    N, D = X.shape
    centers = np.empty((K, D), dtype=np.float32)
    # first center
    idx0 = int(rng.integers(0, N))
    centers[0] = X[idx0]
    # distances to nearest center
    d2 = np.sum((X - centers[0]) ** 2, axis=1).astype(np.float64)
    for k in range(1, K):
        s = float(d2.sum())
        if not np.isfinite(s) or s <= 1e-12:
            # all points identical -> random
            centers[k] = X[int(rng.integers(0, N))]
            continue
        probs = d2 / s
        idx = int(rng.choice(N, p=probs))
        centers[k] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centers[k]) ** 2, axis=1))
    return centers


def _kmeans_assign_labels(X: np.ndarray, C: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Assign each row of X to nearest center in C (squared Euclidean)."""
    N = X.shape[0]
    K = C.shape[0]
    labels = np.empty((N,), dtype=np.int64)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Xc = X[s:e]  # [b,D]
        # [b,K]
        d2 = ((Xc[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels[s:e] = d2.argmin(axis=1).astype(np.int64)
    return labels


def _kmeans_lloyd(
    X: np.ndarray,
    K: int,
    iters: int = 20,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means (Lloyd). Returns (labels[N], centers[K,D])."""
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    rng = np.random.default_rng(int(seed))
    C = _kmeans_pp_init(X, K, rng)

    labels = _kmeans_assign_labels(X, C)
    for _ in range(int(iters)):
        prev = labels
        # update centers
        C_new = np.zeros((K, D), dtype=np.float32)
        counts = np.bincount(labels, minlength=K).astype(np.int64)
        np.add.at(C_new, labels, X)
        nonempty = counts > 0
        C_new[nonempty] /= counts[nonempty][:, None].astype(np.float32)

        # reinit empty clusters to far points (helps stability)
        if not np.all(nonempty):
            # distance to nearest center among non-empty
            C_ref = C_new[nonempty]
            d2 = ((X[:, None, :] - C_ref[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            far_order = np.argsort(-d2)
            empties = np.nonzero(~nonempty)[0]
            for t, k in enumerate(empties.tolist()):
                C_new[k] = X[int(far_order[t % N])]

        C = C_new
        labels = _kmeans_assign_labels(X, C)

        if np.array_equal(labels, prev):
            break

    return labels.astype(np.int64), C.astype(np.float32)


def _davies_bouldin_index(X: np.ndarray, labels: np.ndarray, C: np.ndarray, eps: float = 1e-6) -> float:
    """Davies–Bouldin index (lower is better)."""
    X = np.asarray(X, dtype=np.float32)
    labels = labels.astype(np.int64)
    K = C.shape[0]

    # scatter per cluster: mean distance to centroid
    scat = np.zeros((K,), dtype=np.float32)
    counts = np.bincount(labels, minlength=K).astype(np.int64)
    for k in range(K):
        if counts[k] <= 0:
            scat[k] = 0.0
            continue
        pts = X[labels == k]
        scat[k] = float(np.sqrt(((pts - C[k]) ** 2).sum(axis=1)).mean())

    # centroid distances
    # [K,K]
    D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=2) + eps).astype(np.float32)
    np.fill_diagonal(D, np.inf)

    # R_kl
    R = (scat[:, None] + scat[None, :]) / D
    db = float(np.max(R, axis=1).mean())
    if not np.isfinite(db):
        db = float("inf")
    return db


def _build_spatial_adj(coords_xy: np.ndarray, knn_k: int, max_edge_dist_px: float) -> List[List[int]]:
    """Build undirected adjacency from kNN + distance threshold (same as old logic)."""
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    N = coords_xy.shape[0]
    tree = cKDTree(coords_xy)
    k = min(int(knn_k) + 1, N)
    dists, nbrs = tree.query(coords_xy, k=k, workers=-1)
    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]

    adj: List[List[int]] = [[] for _ in range(N)]
    thr = float(max_edge_dist_px)
    for i in range(N):
        ok = dists[i] <= thr
        neigh = nbrs[i][ok].tolist()
        adj[i].extend(neigh)
    for i in range(N):
        for j in adj[i]:
            adj[j].append(i)
    return adj


def _split_by_connectivity(labels: np.ndarray, adj: List[List[int]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Ensure each cluster is spatially connected; split disconnected components."""
    labels = labels.astype(np.int64)
    N = labels.shape[0]
    new_id = -np.ones((N,), dtype=np.int64)
    members: List[np.ndarray] = []
    cid = 0

    # group nodes by original label
    for k in np.unique(labels).tolist():
        nodes = np.nonzero(labels == k)[0].astype(np.int64)
        if nodes.size == 0:
            continue
        node_set = set(nodes.tolist())
        visited = set()
        for seed in nodes.tolist():
            if seed in visited:
                continue
            # BFS component
            q = [seed]
            comp = []
            while q:
                v = q.pop()
                if v in visited:
                    continue
                visited.add(v)
                comp.append(v)
                for u in adj[v]:
                    if u in node_set and u not in visited:
                        q.append(u)
            comp = np.array(sorted(comp), dtype=np.int64)
            new_id[comp] = cid
            members.append(comp)
            cid += 1

    if (new_id < 0).any():
        # should not happen, but keep safe
        un = np.nonzero(new_id < 0)[0].astype(np.int64)
        for v in un.tolist():
            new_id[v] = cid
            members.append(np.array([v], dtype=np.int64))
            cid += 1

    return new_id, members


def _spatial_majority_smooth(
    labels: np.ndarray,
    adj: List[List[int]],
    iters: int = 2,
) -> np.ndarray:
    """Simple spatial smoothing: each node takes majority label of its neighbors (+self) for a few iterations.

    This is a lightweight way to reduce 'salt-and-pepper' scattered clusters when K is small (e.g. 4-5).
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    N = labels.shape[0]
    if N == 0:
        return labels
    iters = max(0, int(iters))
    for _ in range(iters):
        new = labels.copy()
        for i in range(N):
            neigh = adj[i]
            if not neigh:
                continue
            vals = [labels[i]] + [labels[j] for j in neigh]
            # majority vote
            uniq, cnt = np.unique(np.array(vals, dtype=np.int64), return_counts=True)
            new[i] = int(uniq[int(cnt.argmax())])
        labels = new
    return labels


def build_small_spatial_clusters(
    coords_xy: np.ndarray,
    expr_sum: Optional[np.ndarray] = None,
    target_size: int = 15,
    max_size: int = 25,  # kept for backward-compat; no longer used as a hard cap
    knn_k: int = 20,
    max_edge_dist_px: float = 120.0,
    expr_weight: float = 1.0,
    candidate_cells_per_cluster: Optional[List[int]] = None,
    kmeans_iters: int = 15,
    seed: int = 7,
    # ===== NEW: coarse clustering controls =====
    k_range: Optional[Tuple[int, int]] = (4, 8),
    fixed_k: Optional[int] = None,
    smooth_iters: int = 2,
    enforce_connectivity: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    ✅ cluster 逻辑（用于 CMI 的离散均匀抽样）

    你现在的需求是“只要 4-5 类，且不要太散”。因此这里新增了 **coarse clustering** 模式：
    - 默认在 K∈[4,5] 里自动选一个（用 Davies–Bouldin index，越小越好）
    - 可选进行少量 **空间多数投票平滑**（减少散点）
    - 默认 **不做连通分量拆分**（拆分会把簇数越拆越多，与你的目标冲突）
      如确实需要严格空间连通，可设 enforce_connectivity=True

    仍保留旧接口参数（target_size/candidate_cells_per_cluster），以兼容以前调用。
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    N = int(coords_xy.shape[0])
    if N <= 1:
        cid = np.zeros((N,), dtype=np.int64)
        mem = [np.arange(N, dtype=np.int64)]
        return cid, mem

    if expr_sum is None:
        expr_sum = np.zeros((N,), dtype=np.float32)
    expr_sum = np.asarray(expr_sum, dtype=np.float32).reshape(-1)
    if expr_sum.shape[0] != N:
        raise ValueError(f"expr_sum length mismatch: expect {N}, got {expr_sum.shape[0]}")

    # joint feature: [x, y, expr_sum]
    X = np.concatenate([coords_xy, expr_sum[:, None]], axis=1).astype(np.float32)
    Xn, _, _ = _standardize_features(X)
    Xn[:, 2] *= float(expr_weight)

    # build adjacency once (used by smoothing / connectivity)
    adj = _build_spatial_adj(coords_xy, knn_k=knn_k, max_edge_dist_px=max_edge_dist_px)

    # -------- choose K candidates --------
    if fixed_k is not None:
        K_candidates = [int(fixed_k)]
    elif k_range is not None:
        kmin, kmax = int(k_range[0]), int(k_range[1])
        if kmax < kmin:
            kmin, kmax = kmax, kmin
        kmin = max(2, kmin)
        kmax = max(2, kmax)
        kmax = min(kmax, max(2, N - 1))
        K_candidates = list(range(kmin, kmax + 1))
    else:
        # (legacy) candidate K derived from neighborhood size
        if candidate_cells_per_cluster is None:
            base = max(5, int(target_size))
            cells_list = [
                max(5, int(round(base * 0.6))),
                max(5, int(round(base * 0.8))),
                base,
                int(round(base * 1.25)),
                int(round(base * 1.6)),
                int(round(base * 2.0)),
            ]
        else:
            cells_list = [max(2, int(x)) for x in candidate_cells_per_cluster]
        cells_list = sorted({int(x) for x in cells_list if int(x) >= 2})
        K_candidates = sorted({max(2, min(N - 1, int(round(N / float(c))))) for c in cells_list})
        K_candidates = [k for k in K_candidates if 2 <= k <= min(N - 1, 2048)]
        if not K_candidates:
            K_candidates = [max(2, min(N - 1, int(round(N / max(target_size, 2)))))]

    # dedup + guard
    K_candidates = sorted({int(k) for k in K_candidates if 2 <= int(k) <= max(2, N - 1)})
    if not K_candidates:
        K_candidates = [max(2, min(N - 1, 4))]

    best = {"db": float("inf"), "labels": None, "centers": None, "K": None}

    for K in K_candidates:
        labels, centers = _kmeans_lloyd(Xn, K=int(K), iters=int(kmeans_iters), seed=int(seed))
        db = _davies_bouldin_index(Xn, labels, centers)

        # penalty: too many tiny clusters usually indicates over-splitting
        counts = np.bincount(labels, minlength=int(K)).astype(np.int64)
        tiny = float(np.mean(counts < 3))
        score = float(db + 0.25 * tiny)

        if score < best["db"]:
            best = {"db": score, "labels": labels, "centers": centers, "K": int(K)}

    labels_best = best["labels"]
    if labels_best is None:
        raise RuntimeError("k-means failed to produce clustering.")

    # -------- smooth to reduce scattered labels --------
    labels_best = _spatial_majority_smooth(labels_best, adj, iters=int(smooth_iters))

    if enforce_connectivity:
        # NOTE: may increase the number of clusters, so default is False
        cluster_id, members = _split_by_connectivity(labels_best, adj)
        return cluster_id.astype(np.int64), members

    # build members WITHOUT splitting, but ensure cluster_id is contiguous: 0..K-1
    cluster_id_raw = labels_best.astype(np.int64)

    uniq = np.unique(cluster_id_raw).astype(np.int64)
    # remap old labels -> 0..K-1
    remap = {int(old): int(new) for new, old in enumerate(uniq.tolist())}

    cluster_id = np.empty_like(cluster_id_raw)
    for old, new in remap.items():
        cluster_id[cluster_id_raw == old] = new

    K = int(len(uniq))
    members: List[np.ndarray] = []
    for k in range(K):
        idx = np.nonzero(cluster_id == k)[0].astype(np.int64)
        members.append(idx)

    return cluster_id.astype(np.int64), members


# ============================================================
# 2) Patch crop & compute z_i / H_i (H_i = union of all cells)
# ============================================================

def _crop_square_rgb(he: np.ndarray, cx: float, cy: float, ps: int) -> np.ndarray:
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


def _crop_square_mask(mask: np.ndarray, cx: float, cy: float, ps: int) -> np.ndarray:
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


def compute_z_H_allcells(
    he: np.ndarray,
    mask: np.ndarray,
    coords_xy: np.ndarray,
    encoder: HEEncoderWrapper,
    device: torch.device,
    patch_size_px: int = 100,
    batch_size: int = 64,
    out_device: str = "cpu",
    out_dtype: torch.dtype = torch.float16,
    amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    z_i: raw patch embedding
    H_i: patch with only union-of-cells pixels kept (mask>0); others set to 0

    ✅ OOM-safe:
    - GPU 上只跑当前 batch 的 encoder forward
    - batch 输出立刻搬到 CPU（避免在 GPU 堆满所有节点 embedding）
    - OOM 自动减小 batch 重试
    """
    ps = int(patch_size_px)
    N = coords_xy.shape[0]

    z_chunks: List[torch.Tensor] = []
    h_chunks: List[torch.Tensor] = []

    z_buf: List[np.ndarray] = []
    h_buf: List[np.ndarray] = []

    cur_bs = int(batch_size)
    out_dev = torch.device(out_device)

    def flush_once() -> None:
        nonlocal z_buf, h_buf
        if not z_buf:
            return

        z_t = torch.from_numpy(np.stack(z_buf, 0)).to(device=device, non_blocking=True)
        h_t = torch.from_numpy(np.stack(h_buf, 0)).to(device=device, non_blocking=True)

        z_emb = encoder(z_t, amp=amp)
        h_emb = encoder(h_t, amp=amp)

        z_chunks.append(z_emb.detach().to(device=out_dev, dtype=out_dtype, non_blocking=True))
        h_chunks.append(h_emb.detach().to(device=out_dev, dtype=out_dtype, non_blocking=True))

        del z_t, h_t, z_emb, h_emb
        z_buf.clear()
        h_buf.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    i = 0
    while i < N:
        cx, cy = float(coords_xy[i, 0]), float(coords_xy[i, 1])
        rgb = _crop_square_rgb(he, cx, cy, ps)
        msk = _crop_square_mask(mask, cx, cy, ps)

        z_buf.append(rgb)
        keep = (msk > 0)
        h_rgb = rgb.copy()
        h_rgb[~keep] = 0
        h_buf.append(h_rgb)

        if len(z_buf) >= cur_bs or i == N - 1:
            try:
                flush_once()
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                if cur_bs <= 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                continue

        i += 1

    z_all = torch.cat(z_chunks, dim=0)
    h_all = torch.cat(h_chunks, dim=0)
    return z_all, h_all


# =========================================================
# NOTE (2026-02): Node-centric HE embedding
#
# Updated design (user request):
#   - Each node/cell uses a square HE patch centered at its centroid (coords).
#   - The node HE feature H_i is the embedding of that patch.
#   - z is the list of per-cell embeddings produced in the same way.
#
# Implementation detail:
#   - We reuse compute_z_only_allcells() (square crop, no mask) to compute node_he_feat.
#   - In the causal module, we set z_local_feat = node_he_feat and h_feat = node_he_feat.
# =========================================================



# ============================================================
# 2.5) Multi-scale Z: local + global (stitched 3x3 tiles)
#      and orthogonalized local-beyond-global residual
# ============================================================

def _parse_xy_from_sample_id(sample_id: str) -> Tuple[int, int]:
    """Parse folder name 'x_y' -> (x, y)."""
    parts = str(sample_id).split("_")
    if len(parts) != 2:
        raise ValueError(f"Expect sample_id as 'x_y', got: {sample_id}")
    return int(parts[0]), int(parts[1])



def load_global_he_image(path: str | Path) -> np.ndarray:
    """Load a pre-built global HE mosaic from .npy or .npz (expects RGB [H,W,3])."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"global_he_path not found: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
    elif p.suffix.lower() == ".npz":
        npz = np.load(str(p))
        # prefer common keys, else first array
        key = None
        for k in ["he", "img", "image", "mosaic", "he_global"]:
            if k in npz.files:
                key = k
                break
        if key is None:
            key = npz.files[0]
        arr = npz[key]
    else:
        raise ValueError(f"Unsupported global_he_path suffix: {p.suffix} (only .npy/.npz)")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"global HE must be RGB [H,W,3], got {arr.shape} from {p}")
    return arr


def stitch_global_he_grid(
    sample_dir: Path,
    radius_tiles: int,
    he_filename: str = "HE.npy",
    center_fallback: str = "repeat",  # repeat | zeros
) -> np.ndarray:
    """
    Build a global HE mosaic by stitching a (2r+1)×(2r+1) neighborhood centered at sample_dir (x_y).

    - Tiles are assumed to be saved per-sample as: <sample_dir>/<he_filename>, e.g. HE.npy
    - sample_dir name must be "x_y" (x=col, y=row), y increases downward.

    If a neighbor tile is missing:
      - 'repeat': use center tile
      - 'zeros' : fill zeros

    Returns
    -------
    he_mosaic : np.ndarray [H*(2r+1), W*(2r+1), 3]
    """
    r = int(radius_tiles)
    if r < 0:
        raise ValueError(f"radius_tiles must be >=0, got {r}")

    sample_dir = Path(sample_dir)
    center_path = sample_dir / he_filename
    if not center_path.exists():
        raise FileNotFoundError(f"Center HE file not found: {center_path}")
    he_center = np.load(str(center_path))
    if he_center.ndim != 3 or he_center.shape[2] != 3:
        raise ValueError(f"HE must be RGB [H,W,3], got {he_center.shape}")

    tile_h, tile_w = int(he_center.shape[0]), int(he_center.shape[1])
    cx, cy = _parse_xy_from_sample_id(sample_dir.name)

    grid = 2 * r + 1
    mosaic = np.zeros((tile_h * grid, tile_w * grid, 3), dtype=he_center.dtype)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            nx, ny = cx + dx, cy + dy
            n_dir = sample_dir.parent / f"{nx}_{ny}"
            n_path = n_dir / he_filename

            if n_path.exists():
                he_tile = np.load(str(n_path))
                if he_tile.shape != he_center.shape:
                    raise ValueError(
                        f"Neighbor tile shape mismatch: {n_path} has {he_tile.shape}, "
                        f"expected {he_center.shape}"
                    )
            else:
                if center_fallback == "repeat":
                    he_tile = he_center
                elif center_fallback == "zeros":
                    he_tile = np.zeros_like(he_center)
                else:
                    raise ValueError(f"Unknown center_fallback={center_fallback}")

            oy = (dy + r) * tile_h
            ox = (dx + r) * tile_w
            mosaic[oy:oy + tile_h, ox:ox + tile_w, :] = he_tile

    return mosaic

def compute_z_only_allcells(
    he: np.ndarray,
    coords_xy: np.ndarray,
    encoder: HEEncoderWrapper,
    device: torch.device,
    patch_size_px: int,
    batch_size: int = 64,
    out_device: str = "cpu",
    out_dtype: torch.dtype = torch.float16,
    amp: bool = True,
) -> torch.Tensor:
    """
    Compute per-cell patch embeddings z_i only (no mask), OOM-safe.
    Output is moved to out_device immediately.
    """
    ps = int(patch_size_px)
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    N = int(coords_xy.shape[0])

    chunks: List[torch.Tensor] = []
    buf: List[np.ndarray] = []
    cur_bs = int(batch_size)
    out_dev = torch.device(out_device)

    def flush_once():
        nonlocal buf
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf, 0)).to(device=device, non_blocking=True)
        emb = encoder(t, amp=amp)
        chunks.append(emb.detach().to(device=out_dev, dtype=out_dtype, non_blocking=True))
        del t, emb
        buf.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    i = 0
    while i < N:
        cx, cy = float(coords_xy[i, 0]), float(coords_xy[i, 1])
        rgb = _crop_square_rgb(he, cx, cy, ps)
        buf.append(rgb)

        if len(buf) >= cur_bs or i == N - 1:
            try:
                flush_once()
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                if cur_bs <= 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                continue

        i += 1

    return torch.cat(chunks, dim=0)


def orthogonalize_local_by_global(
    local_feat: torch.Tensor,
    global_feat: torch.Tensor,
    ridge_lambda: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Orthogonalize local confounder by removing the part explained by global:

        L_tilde = L - E[L | G]

    We fit a (ridge) linear model across cells in the sample:
        L ≈ G W + b

    local_feat/global_feat are expected on CPU (proj_device), float32/float16.
    Returns L_tilde on the same device as inputs.
    """
    if local_feat.shape[0] != global_feat.shape[0]:
        raise ValueError(f"N mismatch: local={local_feat.shape}, global={global_feat.shape}")
    dev = local_feat.device
    L = local_feat.detach().float().cpu().numpy().astype(np.float64)  # [N,dl]
    G = global_feat.detach().float().cpu().numpy().astype(np.float64) # [N,dg]
    N, dg = G.shape
    dl = L.shape[1]

    # add intercept
    G1 = np.concatenate([G, np.ones((N, 1), dtype=np.float64)], axis=1)  # [N, dg+1]
    lam = float(ridge_lambda)
    A = (G1.T @ G1) + lam * np.eye(dg + 1, dtype=np.float64)
    B = (G1.T @ L)
    W = np.linalg.solve(A, B)  # [dg+1, dl]
    L_pred = G1 @ W
    L_res = (L - L_pred).astype(np.float32)

    out = torch.from_numpy(L_res).to(device=dev)
    stats = {
        "ridge_lambda": float(lam),
        "dg": float(dg),
        "dl": float(dl),
    }
    return out, stats


# ============================================================
# 3) LR mapping and expression scalar
# ============================================================

def build_lr_map(gene_names: List[str], pairs_kept: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[int]]]:
    name_to_idx = {str(g): i for i, g in enumerate(gene_names)}
    out: Dict[int, Dict[str, List[int]]] = {}
    for d in pairs_kept:
        pid = int(d["pair_id"])
        lig_parts = [name_to_idx[str(x)] for x in d["ligand_parts"]]
        rec_parts = [name_to_idx[str(x)] for x in d["receptor_parts"]]
        out[pid] = {"lig_idx": lig_parts, "rec_idx": rec_parts}
    return out


def cell_expr_scalar(expr_log: torch.Tensor, cell_ids: torch.Tensor, gene_idx: List[int]) -> torch.Tensor:
    if len(gene_idx) == 1:
        return expr_log[cell_ids, gene_idx[0]]
    x = expr_log[cell_ids][:, gene_idx]
    return torch.min(x, dim=1).values


# ============================================================
# 4) Gaussian CMI (with analytic expectation over discrete-uniform cluster RVs)
# ============================================================

def _shrink_cov(cov: np.ndarray, lam: float) -> np.ndarray:
    D = cov.shape[0]
    tr = float(np.trace(cov))
    scale = tr / max(D, 1)
    return (1.0 - lam) * cov + lam * (scale * np.eye(D, dtype=cov.dtype))


def _fit_gaussian_from_weighted_moments(
    sum_w: float,
    sum_wx: np.ndarray,
    sum_wxx: np.ndarray,
    shrinkage: float
) -> Tuple[np.ndarray, np.ndarray]:
    mu = sum_wx / max(sum_w, 1e-12)
    M2 = sum_wxx / max(sum_w, 1e-12)
    cov = M2 - np.outer(mu, mu)
    cov = _shrink_cov(cov, lam=shrinkage)
    cov = cov + 1e-6 * np.eye(cov.shape[0], dtype=cov.dtype)
    return mu.astype(np.float64), cov.astype(np.float64)


def _chol_inv_logdet(cov: np.ndarray) -> Tuple[np.ndarray, float]:
    L = np.linalg.cholesky(cov)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L) + 1e-12)))
    inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(cov.shape[0], dtype=cov.dtype)))
    return inv, logdet


def _cond_params(mu: np.ndarray, cov: np.ndarray, idx_u: np.ndarray, idx_v: np.ndarray):
    mu_u = mu[idx_u]
    mu_v = mu[idx_v]

    Suu = cov[np.ix_(idx_u, idx_u)]
    Suv = cov[np.ix_(idx_u, idx_v)]
    Svv = cov[np.ix_(idx_v, idx_v)]

    inv_v, _ = _chol_inv_logdet(Svv)
    A = Suv @ inv_v
    cov_cond = Suu - A @ Suv.T
    cov_cond = cov_cond + 1e-6 * np.eye(cov_cond.shape[0], dtype=cov_cond.dtype)
    inv_cond, logdet_cond = _chol_inv_logdet(cov_cond)
    return mu_u, mu_v, A, inv_cond, logdet_cond


def _avg_logpdf_cond_constV(
    u_mean: np.ndarray,
    u_cov: np.ndarray,
    v: np.ndarray,
    mu_u: np.ndarray,
    mu_v: np.ndarray,
    A: np.ndarray,
    inv_cond: np.ndarray,
    logdet_cond: float,
) -> float:
    du = u_mean.shape[0]
    log2pi = math.log(2.0 * math.pi)

    mu_cond = mu_u + A @ (v - mu_v)
    diff = (u_mean - mu_cond).reshape(-1, 1)

    # ✅ 避免 numpy 1.25 的 “ndim>0 to scalar” warning
    quad_mean = float((diff.T @ inv_cond @ diff).item())
    quad_var = float(np.trace(inv_cond @ u_cov))

    const = -0.5 * (du * log2pi + logdet_cond)
    return const - 0.5 * (quad_mean + quad_var)


def _avg_logpdf_cond_ZY_randomY(
    u_mean: np.ndarray,
    u_cov: np.ndarray,
    z: np.ndarray,
    y_mean: float,
    y_var: float,
    mu_u: np.ndarray,
    mu_v: np.ndarray,
    A: np.ndarray,
    inv_cond: np.ndarray,
    logdet_cond: float,
    y_index_in_v: int,
    z_dim: int,
) -> float:
    du = u_mean.shape[0]
    log2pi = math.log(2.0 * math.pi)

    A_y = A[:, y_index_in_v].reshape(-1, 1)
    v_const = mu_v.copy()
    v_const[:] = 0.0
    v_const[:z_dim] = z
    v_const[y_index_in_v] = 0.0

    mu_const = mu_u + A @ (v_const - mu_v)
    mean_r = (u_mean - mu_const).reshape(-1, 1) - A_y * float(y_mean)
    cov_r = u_cov + (A_y @ A_y.T) * float(y_var)

    quad_mean = float((mean_r.T @ inv_cond @ mean_r).item())
    quad_var = float(np.trace(inv_cond @ cov_r))

    const = -0.5 * (du * log2pi + logdet_cond)
    return const - 0.5 * (quad_mean + quad_var)


# ============================================================
# 4.5) CMI score normalization to [0,1] (允许负数，不丢弃)
# ============================================================

def normalize_to_01(
    x: np.ndarray,
    clip_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize arbitrary real-valued vector -> [0,1], keeping negatives.

    Default:
      - robust clip by percentiles (P1,P99) to reduce rare outliers (still keeps negatives)
      - min-max to [0,1]

    If you want strict min-max (no percentile clip), set clip_percentiles=None.
    """
    v = np.asarray(x, dtype=np.float32).copy()
    if v.size == 0:
        return v.astype(np.float32), {"clip_lo": 0.0, "clip_hi": 0.0}

    if clip_percentiles is not None:
        lo_p, hi_p = float(clip_percentiles[0]), float(clip_percentiles[1])
        lo = float(np.percentile(v, lo_p))
        hi = float(np.percentile(v, hi_p))
        if hi < lo:
            hi, lo = lo, hi
        v = np.clip(v, lo, hi)
    else:
        lo = float(v.min())
        hi = float(v.max())

    denom = max(hi - lo, eps)
    v01 = (v - lo) / denom
    v01 = np.clip(v01, 0.0, 1.0).astype(np.float32)

    stats = {
        "clip_lo": float(lo),
        "clip_hi": float(hi),
        "min_after": float(v01.min()),
        "max_after": float(v01.max()),
    }
    return v01, stats


# ============================================================
# 5) Main builder
# ============================================================

class CausalGraphBuilder(nn.Module):
    """
    forward(sample_dict) -> {"graph": Graph, "debug": ...}
    """
    def __init__(
        self,
        he_encoder: HEEncoderWrapper,
        enc_out_dim: int,
        extra_he_dim: int = 0,
        device: str = "cuda",
        proj_device: str = "cpu",
        z_proj_dim: int = 16,
        h_proj_dim: int = 16,
        shrinkage: float = 1e-3,
        local_global_ridge: float = 1e-3,
        min_edges_per_lr: int = 20,
        # ✅ cmi normalization
        cmi_clip_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
        return_debug: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.proj_device = torch.device(proj_device)

        self.he_encoder = he_encoder.to(self.device)

        self.shrinkage = float(shrinkage)
        self.local_global_ridge = float(local_global_ridge)
        self.min_edges_per_lr = int(min_edges_per_lr)
        self.cmi_clip_percentiles = cmi_clip_percentiles
        self.return_debug = bool(return_debug)

        # ✅ no LazyLinear
        torch.manual_seed(7)
        in_dim_total = int(enc_out_dim) + int(extra_he_dim)
        self.z_proj = nn.Linear(int(in_dim_total), int(z_proj_dim), bias=False).to(self.proj_device)
        self.h_proj = nn.Linear(int(in_dim_total), int(h_proj_dim), bias=False).to(self.proj_device)
        for p in self.z_proj.parameters():
            p.requires_grad_(False)
        for p in self.h_proj.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def forward(
        self,
        sample: Dict[str, Any],
        # clustering params
        cluster_target: int = 15,
        cluster_max: int = 25,
        cluster_knn: int = 20,
        cluster_max_edge_dist_px: float = 120.0,
        # ===== NEW: coarse clustering (want 4-5 classes, less scattered) =====
        cluster_k_min: int = 4,
        cluster_k_max: int = 8,
        cluster_smooth_iters: int = 2,
        cluster_enforce_connectivity: bool = False,
        # ===== multi-scale Z_i params (config-driven) =====
        mpp: float = 0.5,                 # micron / pixel
        local_um: float = 250.0,           # local physical window
        global_um: float = 1000.0,          # global physical window
        # NEW: node-centric patch for per-cell embedding (H_i and z list)
        # Default ~80um covers one cell + immediate morphology; mpp=0.5 => 160px.
        node_patch_um: float = 80.0,
        tile_size_px: Optional[int] = None,# HE tile size (e.g. 512)
        global_center_fallback: str = "repeat",
        global_he_path: Optional[str] = None,
        # explicit pixel overrides (highest priority)
        local_patch_size_px: Optional[int] = None,
        global_patch_size_px: Optional[int] = None,
        node_patch_size_px: Optional[int] = None,
        patch_batch_size: int = 64,
        # encoder params
        amp: bool = True,
    ) -> Dict[str, Any]:

        # ====================================================
        # 0) Basic fetch
        # ====================================================
        he = sample["he"]                      # np [H,W,3]
        mask = sample["mask"]                  # np [H,W]
        coords = sample["coords"].to(self.device)      # [N,2]
        expr_log = sample["expr_log"].to(self.device)  # [N,G]
        edge_index = sample["edge_index"].to(self.device).long()
        edge_attr = sample["edge_attr"].to(self.device).float()
        gene_names = sample["gene_names"]
        pairs_kept = sample["lr_pairs"]["pairs_kept"]

        N = coords.shape[0]
        E = edge_index.shape[1]

        coords_np = self._np(coords)
        expr_sum_np = self._np(expr_log.sum(dim=1))

        # ====================================================
        # 1) Small spatial clusters (for CMI RVs)
        # ====================================================
        cluster_id, cluster_members = build_small_spatial_clusters(
            coords_np,
            expr_sum=expr_sum_np,
            target_size=cluster_target,
            max_size=cluster_max,
            knn_k=cluster_knn,
            max_edge_dist_px=cluster_max_edge_dist_px,
            # coarse clustering: force 4-5 clusters by default
            k_range=(int(cluster_k_min), int(cluster_k_max)),
            smooth_iters=int(cluster_smooth_iters),
            enforce_connectivity=bool(cluster_enforce_connectivity),
        )

        # ====================================================
        # 2) Multi-scale window sizes  (*** FIXED ORDER ***)
        # ====================================================
        mpp_f = float(mpp)
        if mpp_f <= 0:
            raise ValueError(f"mpp must be >0, got {mpp_f}")

        tile_h, tile_w = int(he.shape[0]), int(he.shape[1])

        if tile_size_px is not None:
            tile_size_px = int(tile_size_px)
            if tile_size_px != tile_h or tile_size_px != tile_w:
                raise ValueError(
                    f"tile_size_px({tile_size_px}) must match sample['he'] size ({tile_w}x{tile_h})."
                )

        # ---- local / global / node patch sizes (px) ----
        lp = int(local_patch_size_px) if local_patch_size_px is not None else int(round(local_um / mpp_f))
        gp_req = int(global_patch_size_px) if global_patch_size_px is not None else int(round(global_um / mpp_f))
        np_ps = int(node_patch_size_px) if node_patch_size_px is not None else int(round(float(node_patch_um) / mpp_f))

        lp = max(1, lp)
        gp_req = max(1, gp_req)
        np_ps = max(1, np_ps)

        # symmetric crop prefers even sizes
        if (lp % 2) == 1:
            lp += 1
        if (gp_req % 2) == 1:
            gp_req += 1
        if (np_ps % 2) == 1:
            np_ps += 1

        # ====================================================
        # 3) NEW: node-centric embedding for each cell
        #    - H_i: square patch around centroid -> encoder embedding
        #    - z: list of per-cell embeddings (same as H_i)
        # ====================================================
        node_he_feat = compute_z_only_allcells(
            he=he,
            coords_xy=coords_np,
            encoder=self.he_encoder,
            device=self.device,
            patch_size_px=np_ps,
            batch_size=patch_batch_size,
            out_device=str(self.proj_device),
            out_dtype=torch.float16,
            amp=amp,
        )
        # concat traditional morphology features if provided by main
        morph_feat_cpu = sample.get("morph_feat_cpu", None)
        if morph_feat_cpu is not None:
            if isinstance(morph_feat_cpu, torch.Tensor):
                mf = morph_feat_cpu.to(device=node_he_feat.device, non_blocking=True).to(dtype=node_he_feat.dtype)
            else:
                mf = torch.from_numpy(np.asarray(morph_feat_cpu)).to(device=node_he_feat.device, non_blocking=True).to(dtype=node_he_feat.dtype)
            if mf.ndim != 2 or mf.shape[0] != node_he_feat.shape[0]:
                raise ValueError(f"morph_feat shape mismatch: {getattr(mf,'shape',None)} vs node_he_feat {tuple(node_he_feat.shape)}")
            node_he_feat = torch.cat([node_he_feat, mf], dim=1)
        z_local_feat = node_he_feat
        h_feat = node_he_feat

        # ====================================================
        # 4) GLOBAL mosaic stitching
        # ====================================================
        paths = sample.get("paths", {})
        r_for_msg: Optional[int] = None  # only used when stitching

        # If user provides a pre-built global mosaic, use it directly.
        # Assumption: current patch HE (sample["he"]) is the CENTER tile of that mosaic.
        # Therefore, we can recover the coordinate shift by centering:
        #   offset = (mosaic_size - tile_size) / 2
        if global_he_path is not None and str(global_he_path).strip() != "":
            he_global = load_global_he_image(str(global_he_path))

            ox = (float(he_global.shape[1]) - float(tile_w)) / 2.0
            oy = (float(he_global.shape[0]) - float(tile_h)) / 2.0
            ox_i = int(round(ox))
            oy_i = int(round(oy))
            if abs(ox - ox_i) > 1e-3 or abs(oy - oy_i) > 1e-3:
                raise ValueError(
                    f"global mosaic size {he_global.shape[:2]} is not centered w.r.t tile "
                    f"size {(tile_h, tile_w)} (non-integer center offset: ({ox}, {oy}))."
                )
            if ox_i < 0 or oy_i < 0:
                raise ValueError(
                    f"global mosaic size {he_global.shape[:2]} is smaller than tile "
                    f"size {(tile_h, tile_w)}."
                )

            coords_global = coords_np.copy()
            coords_global[:, 0] += float(ox_i)
            coords_global[:, 1] += float(oy_i)

        else:
            # Fall back to the original logic: stitch a (2r+1)x(2r+1) neighborhood grid
            sample_dir_str = paths.get("sample_dir", None)
            if sample_dir_str is None:
                raise KeyError("sample['paths']['sample_dir'] is required for global HE stitching.")
            sample_dir = Path(sample_dir_str)

            # choose radius so stitched mosaic covers global patch
            # mosaic side = tile_w * (2r+1) >= gp_req
            r = int(math.ceil((float(gp_req) / float(tile_w) - 1.0) / 2.0))
            r = max(0, r)
            r_for_msg = r

            he_global = stitch_global_he_grid(
                sample_dir,
                radius_tiles=r,
                he_filename="HE.npy",
                center_fallback=str(global_center_fallback),
            )

            # coords shift into global mosaic
            coords_global = coords_np.copy()
            coords_global[:, 0] += float(r * tile_w)
            coords_global[:, 1] += float(r * tile_h)
        gp = int(gp_req)
        min_side = int(min(he_global.shape[0], he_global.shape[1]))
        if gp > min_side:
            raise ValueError(
                f"global_patch_size_px={gp} > global HE side={min_side}. "
                + (f"(stitched radius_tiles={r_for_msg}, tile_size={tile_w})." if r_for_msg is not None else "(using provided global mosaic).")
            )

        z_global_feat = compute_z_only_allcells(
            he=he_global,
            coords_xy=coords_global,
            encoder=self.he_encoder,
            device=self.device,
            patch_size_px=gp,
            batch_size=patch_batch_size,
            out_device=str(self.proj_device),
            out_dtype=torch.float16,
            amp=amp,
        )

        # ====================================================
        # 5) Projection + orthogonalization (local-beyond-global)
        # ====================================================
        
        # ====================================================
        # (morph) concat traditional cell morphology features if provided
        # ====================================================
        morph_feat = sample.get("morph_feat", None)
        if isinstance(morph_feat, torch.Tensor) and morph_feat.numel() > 0:
            mf_local = morph_feat.to(device=z_local_feat.device, dtype=z_local_feat.dtype)
            mf_global = morph_feat.to(device=z_global_feat.device, dtype=z_global_feat.dtype)
            z_local_feat = torch.cat([z_local_feat, mf_local], dim=1)
            z_global_feat = torch.cat([z_global_feat, mf_global], dim=1)
            # If h_feat is pure image embedding (same dim as original z_local before concat), also concat morph
            if h_feat.shape[1] == (z_local_feat.shape[1] - mf_local.shape[1]):
                h_feat = torch.cat([h_feat, mf_local.to(device=h_feat.device, dtype=h_feat.dtype)], dim=1)

        # Safety: if dims still mismatch projection layers, pad/truncate to expected in_features
        def _match_in_dim(x: torch.Tensor, layer: torch.nn.Linear) -> torch.Tensor:
            in_dim = int(layer.in_features)
            if x.shape[1] == in_dim:
                return x
            if x.shape[1] < in_dim:
                pad = torch.zeros((x.shape[0], in_dim - x.shape[1]), device=x.device, dtype=x.dtype)
                return torch.cat([x, pad], dim=1)
            return x[:, :in_dim]

        z_local_feat = _match_in_dim(z_local_feat, self.z_proj)
        z_global_feat = _match_in_dim(z_global_feat, self.z_proj)
        h_feat = _match_in_dim(h_feat, self.h_proj)

        l_low = self.z_proj(z_local_feat.float())   # L_i
        g_low = self.z_proj(z_global_feat.float())  # G_i
        h_low = self.h_proj(h_feat.float())         # H_i

        l_tilde, ortho_stats = orthogonalize_local_by_global(
            local_feat=l_low,
            global_feat=g_low,
            ridge_lambda=self.local_global_ridge,
        )

        # fused Z used in CMI
        z_low = torch.cat([g_low, l_tilde], dim=1)

        # ====================================================
        # 6) (后续 CMI / I1 I2 I3 逻辑完全不变)
        # ====================================================
        lr_map = build_lr_map(gene_names, pairs_kept)

        src = edge_index[0].long()
        dst = edge_index[1].long()
        lr_id = edge_attr[:, 2].long()

        x_mean = np.zeros((E,), dtype=np.float64)
        x_var  = np.zeros((E,), dtype=np.float64)
        y_mean = np.zeros((E,), dtype=np.float64)
        y_var  = np.zeros((E,), dtype=np.float64)
        weight = np.zeros((E,), dtype=np.float64)

        zE = self._np(z_low[dst.cpu() if z_low.device.type == "cpu" else dst])
        hE = self._np(h_low[dst.cpu() if h_low.device.type == "cpu" else dst])

        vec_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

        def get_cluster_vec(c_id: int, pid: int, role: str) -> torch.Tensor:
            key = (int(c_id), int(pid), role)
            if key in vec_cache:
                return vec_cache[key]
            cells = cluster_members[int(c_id)]
            if pid not in lr_map:
                v = torch.zeros((0,), device=self.device)
            else:
                gene_idx = lr_map[pid]["lig_idx"] if role == "lig" else lr_map[pid]["rec_idx"]
                ids = torch.from_numpy(cells).to(self.device, dtype=torch.long)
                v = cell_expr_scalar(expr_log, ids, gene_idx)
            vec_cache[key] = v
            return v

        for e in range(E):
            pid = int(lr_id[e].item())
            j = int(src[e].item())
            i = int(dst[e].item())

            cj = int(cluster_id[j])
            ci = int(cluster_id[i])

            xv = get_cluster_vec(cj, pid, "lig")  # sender lig expr in cluster
            yv = get_cluster_vec(ci, pid, "rec")  # receiver rec expr in cluster

            ns = int(xv.numel())
            nr = int(yv.numel())
            if ns == 0 or nr == 0:
                weight[e] = 0.0
                continue

            xm = float(xv.mean().item())
            ym = float(yv.mean().item())
            vx = float(((xv - xv.mean()) ** 2).mean().item()) if ns > 1 else 0.0
            vy = float(((yv - yv.mean()) ** 2).mean().item()) if nr > 1 else 0.0

            x_mean[e] = xm
            y_mean[e] = ym
            x_var[e] = vx
            y_var[e] = vy
            weight[e] = float(ns * nr)

        dz = zE.shape[1]
        dh = hE.shape[1]

        
        def accum_joint(edge_ids: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build a per-LR (or global) joint Gaussian by aggregating per-edge Gaussian moments.

            Modes:
              J1: V=[X, Y, Z]          -> I1 = CMI(X;Y | Z)
              J2: V=[X, H, Z]          -> I2 = CMI(X;H | Z)   (H is receiver histology embedding)
              J3: V=[Y, Z, X]          -> I3 = CMI(Y;Z | X)   (user: CMI(x_j,z|x_i), receiver=j -> Y, sender=i -> X)
              J4: V=[H, Z, X]          -> I4 = CMI(H;Z | X)   (user: CMI(H_j,z|x_i), receiver=j -> H, sender=i -> X)
            """
            if mode == "J1":   # [X,Y,Z]
                D = 2 + dz
            elif mode == "J2": # [X,H,Z]
                D = 1 + dh + dz
            elif mode == "J3": # [Y,Z,X]
                D = 2 + dz
            elif mode == "J4": # [H,Z,X]
                D = dh + dz + 1
            else:
                raise ValueError(mode)

            sum_w = 0.0
            sum_wx = np.zeros((D,), dtype=np.float64)
            sum_wxx = np.zeros((D, D), dtype=np.float64)

            for e in edge_ids.tolist():
                w = float(weight[e])
                if w <= 0:
                    continue

                if mode == "J1":
                    m = np.concatenate([[x_mean[e], y_mean[e]], zE[e]], axis=0)
                    C = np.zeros((D, D), dtype=np.float64)
                    C[0, 0] = x_var[e]
                    C[1, 1] = y_var[e]

                elif mode == "J2":
                    m = np.concatenate([[x_mean[e]], hE[e], zE[e]], axis=0)
                    C = np.zeros((D, D), dtype=np.float64)
                    C[0, 0] = x_var[e]

                elif mode == "J3":
                    m = np.concatenate([[y_mean[e]], zE[e], [x_mean[e]]], axis=0)
                    C = np.zeros((D, D), dtype=np.float64)
                    C[0, 0] = y_var[e]
                    C[-1, -1] = x_var[e]

                else:  # J4
                    m = np.concatenate([hE[e], zE[e], [x_mean[e]]], axis=0)
                    C = np.zeros((D, D), dtype=np.float64)
                    C[-1, -1] = x_var[e]

                M2 = C + np.outer(m, m)

                sum_w += w
                sum_wx += w * m
                sum_wxx += w * M2

            return _fit_gaussian_from_weighted_moments(
                sum_w, sum_wx, sum_wxx, shrinkage=self.shrinkage
            )

        # ---- global (all edges) joint gaussians ----
        all_edges = np.arange(E, dtype=np.int64)
        mu1_g, cov1_g = accum_joint(all_edges, "J1")
        mu2_g, cov2_g = accum_joint(all_edges, "J2")
        mu3_g, cov3_g = accum_joint(all_edges, "J3")
        mu4_g, cov4_g = accum_joint(all_edges, "J4")

        lr_np = self._np(lr_id)
        uniq = np.unique(lr_np).astype(np.int64)

        I1 = np.zeros((E,), dtype=np.float32)
        I2 = np.zeros((E,), dtype=np.float32)
        I3 = np.zeros((E,), dtype=np.float32)
        I4 = np.zeros((E,), dtype=np.float32)

        for pid in uniq.tolist():
            idx = np.nonzero(lr_np == pid)[0]
            if idx.size < self.min_edges_per_lr:
                mu1, cov1 = mu1_g, cov1_g
                mu2, cov2 = mu2_g, cov2_g
                mu3, cov3 = mu3_g, cov3_g
                mu4, cov4 = mu4_g, cov4_g
            else:
                mu1, cov1 = accum_joint(idx, "J1")
                mu2, cov2 = accum_joint(idx, "J2")
                mu3, cov3 = accum_joint(idx, "J3")
                mu4, cov4 = accum_joint(idx, "J4")

            # I1: CMI(X;Y|Z)   V=[X,Y,Z]
            idx_x = np.array([0])
            idx_y = np.array([1])
            idx_z = np.arange(2, 2 + dz)
            idx_xy = np.array([0, 1])

            mu_xy, mu_z, A_xy, inv_xy, logdet_xy = _cond_params(mu1, cov1, idx_xy, idx_z)
            mu_x,  mu_z2, A_x,  inv_x,  logdet_x  = _cond_params(mu1, cov1, idx_x,  idx_z)
            mu_y,  mu_z3, A_y,  inv_y,  logdet_y  = _cond_params(mu1, cov1, idx_y,  idx_z)

            # I2: CMI(X;H|Z)   V=[X,H,Z]
            idx_X2 = np.array([0])
            idx_H2 = np.arange(1, 1 + dh)
            idx_Z2 = np.arange(1 + dh, 1 + dh + dz)
            idx_XH2 = np.concatenate([idx_X2, idx_H2])

            mu_XH,  mu_Z,  A_XH,  inv_XH,  logdet_XH  = _cond_params(mu2, cov2, idx_XH2, idx_Z2)
            mu_Xo,  mu_Zx, A_Xo,  inv_Xo,  logdet_Xo  = _cond_params(mu2, cov2, idx_X2,  idx_Z2)
            mu_Ho,  mu_Zh, A_Ho,  inv_Ho,  logdet_Ho  = _cond_params(mu2, cov2, idx_H2,  idx_Z2)

            # I3: CMI(Y;Z|X)   V=[Y,Z,X]
            idx_Y3 = np.array([0])
            idx_Z3 = np.arange(1, 1 + dz)
            idx_X3 = np.array([1 + dz])
            idx_YZ3 = np.concatenate([idx_Y3, idx_Z3])

            mu_YZ3, mu_Xc, A_YZ3, inv_YZ3, logdet_YZ3 = _cond_params(mu3, cov3, idx_YZ3, idx_X3)
            mu_Y3,  mu_Xy, A_Y3,  inv_Y3,  logdet_Y3  = _cond_params(mu3, cov3, idx_Y3,  idx_X3)
            mu_Z3,  mu_Xz, A_Z3,  inv_Z3,  logdet_Z3  = _cond_params(mu3, cov3, idx_Z3,  idx_X3)

            # I4: CMI(H;Z|X)   V=[H,Z,X]
            idx_H4 = np.arange(0, dh)
            idx_Z4 = np.arange(dh, dh + dz)
            idx_X4 = np.array([dh + dz])
            idx_HZ4 = np.concatenate([idx_H4, idx_Z4])

            mu_HZ4, mu_X4m, A_HZ4, inv_HZ4, logdet_HZ4 = _cond_params(mu4, cov4, idx_HZ4, idx_X4)
            mu_H4,  mu_X4h, A_H4,  inv_H4,  logdet_H4  = _cond_params(mu4, cov4, idx_H4,  idx_X4)
            mu_Z4,  mu_X4z, A_Z4,  inv_Z4,  logdet_Z4  = _cond_params(mu4, cov4, idx_Z4,  idx_X4)

            for e in idx.tolist():
                if weight[e] <= 0:
                    continue

                z = zE[e]
                h = hE[e]
                mx = float(x_mean[e]); vx = float(x_var[e])
                my = float(y_mean[e]); vy = float(y_var[e])

                # ---- I1 ----
                u_mean_xy = np.array([mx, my], dtype=np.float64)
                u_cov_xy  = np.diag([vx, vy]).astype(np.float64)
                lp_xy = _avg_logpdf_cond_constV(u_mean_xy, u_cov_xy, z, mu_xy, mu_z, A_xy, inv_xy, logdet_xy)

                u_mean_x = np.array([mx], dtype=np.float64)
                u_cov_x  = np.array([[vx]], dtype=np.float64)
                lp_x = _avg_logpdf_cond_constV(u_mean_x, u_cov_x, z, mu_x, mu_z2, A_x, inv_x, logdet_x)

                u_mean_y = np.array([my], dtype=np.float64)
                u_cov_y  = np.array([[vy]], dtype=np.float64)
                lp_y = _avg_logpdf_cond_constV(u_mean_y, u_cov_y, z, mu_y, mu_z3, A_y, inv_y, logdet_y)

                I1[e] = float(lp_xy - lp_x - lp_y)

                # ---- I2 ----
                u_mean_XH = np.concatenate([[mx], h], axis=0).astype(np.float64)
                u_cov_XH  = np.zeros((1 + dh, 1 + dh), dtype=np.float64)
                u_cov_XH[0, 0] = vx
                lp_XH = _avg_logpdf_cond_constV(u_mean_XH, u_cov_XH, z, mu_XH, mu_Z, A_XH, inv_XH, logdet_XH)

                u_mean_Xo = np.array([mx], dtype=np.float64)
                u_cov_Xo  = np.array([[vx]], dtype=np.float64)
                lp_Xo = _avg_logpdf_cond_constV(u_mean_Xo, u_cov_Xo, z, mu_Xo, mu_Zx, A_Xo, inv_Xo, logdet_Xo)

                u_mean_Ho = h.astype(np.float64)
                u_cov_Ho  = np.zeros((dh, dh), dtype=np.float64)
                lp_Ho = _avg_logpdf_cond_constV(u_mean_Ho, u_cov_Ho, z, mu_Ho, mu_Zh, A_Ho, inv_Ho, logdet_Ho)

                I2[e] = float(lp_XH - lp_Xo - lp_Ho)

                # ---- I3 (Y;Z|X) ----
                v_x = np.array([mx], dtype=np.float64)

                u_mean_YZ = np.concatenate([[my], z], axis=0).astype(np.float64)
                u_cov_YZ  = np.zeros((1 + dz, 1 + dz), dtype=np.float64)
                u_cov_YZ[0, 0] = vy
                lp_YZ = _avg_logpdf_cond_constV(u_mean_YZ, u_cov_YZ, v_x, mu_YZ3, mu_Xc, A_YZ3, inv_YZ3, logdet_YZ3)

                u_mean_Y = np.array([my], dtype=np.float64)
                u_cov_Y  = np.array([[vy]], dtype=np.float64)
                lp_Y_only = _avg_logpdf_cond_constV(u_mean_Y, u_cov_Y, v_x, mu_Y3, mu_Xy, A_Y3, inv_Y3, logdet_Y3)

                u_mean_Z = z.astype(np.float64)
                u_cov_Z  = np.zeros((dz, dz), dtype=np.float64)
                lp_Z_only = _avg_logpdf_cond_constV(u_mean_Z, u_cov_Z, v_x, mu_Z3, mu_Xz, A_Z3, inv_Z3, logdet_Z3)

                I3[e] = float(lp_YZ - lp_Y_only - lp_Z_only)

                # ---- I4 (H;Z|X) ----
                u_mean_HZ = np.concatenate([h.astype(np.float64), z.astype(np.float64)], axis=0)
                u_cov_HZ  = np.zeros((dh + dz, dh + dz), dtype=np.float64)
                lp_HZ = _avg_logpdf_cond_constV(u_mean_HZ, u_cov_HZ, v_x, mu_HZ4, mu_X4m, A_HZ4, inv_HZ4, logdet_HZ4)

                u_mean_H = h.astype(np.float64)
                u_cov_H  = np.zeros((dh, dh), dtype=np.float64)
                lp_H_only = _avg_logpdf_cond_constV(u_mean_H, u_cov_H, v_x, mu_H4, mu_X4h, A_H4, inv_H4, logdet_H4)

                u_mean_Z2 = z.astype(np.float64)
                u_cov_Z2  = np.zeros((dz, dz), dtype=np.float64)
                lp_Z_only2 = _avg_logpdf_cond_constV(u_mean_Z2, u_cov_Z2, v_x, mu_Z4, mu_X4z, A_Z4, inv_Z4, logdet_Z4)

                I4[e] = float(lp_HZ - lp_H_only - lp_Z_only2)
        I1_t = torch.from_numpy(I1.astype(np.float32)).to(self.device)  # [E]
        I2_t = torch.from_numpy(I2.astype(np.float32)).to(self.device)  # [E]
        I3_t = torch.from_numpy(I3.astype(np.float32)).to(self.device)  # [E]
        I4_t = torch.from_numpy(I4.astype(np.float32)).to(self.device)  # [E]

        # graph.edge_attr 保持 3 维: [distance_px, coexpr, lr_id]
        if Data is not None:
            graph = Data(num_nodes=int(N), edge_index=edge_index, edge_attr=edge_attr)
        else:
            graph = {"num_nodes": int(N), "edge_index": edge_index, "edge_attr": edge_attr}

        out: Dict[str, Any] = {"graph": graph, "I1": I1_t, "I2": I2_t, "I3": I3_t, "I4": I4_t}
        if self.return_debug:
            out["debug"] = {
                "cluster_id": cluster_id,
                "cluster_sizes": np.array([len(m) for m in cluster_members], dtype=np.int64),
                "I1": I1, "I2": I2, "I3": I3, "I4": I4,
                "z_low": z_low.detach().cpu(),
                "h_low": h_low.detach().cpu(),
                "g_low": g_low.detach().cpu(),
                "l_low": l_low.detach().cpu(),
                "l_tilde": l_tilde.detach().cpu(),
                "ortho": ortho_stats,
                "z_multiscale": {
                    "mpp": float(mpp_f),
                    "local_um": float(local_um),
                    "global_um": float(global_um),
                    "local_px": int(lp),
                    "global_px": int(gp),
                    "radius_tiles": int(r_for_msg) if r_for_msg is not None else 0,
                    "tile_size_px": int(tile_w),
                    "center_fallback": str(global_center_fallback),
                },
            }
        return out
