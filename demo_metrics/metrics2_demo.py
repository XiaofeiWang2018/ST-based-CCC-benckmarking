# %%
"""
============================================================
Contact-based vs Secretion-based CCC Benchmark (Demo Pipeline)
============================================================

GOAL
----
This script benchmarks multiple spatial CCC inference methods using a simple
biophysical prior:
  - Contact-based signaling should be short-range (sender and receiver close).
  - Secretion-based signaling can act over longer distances.
Since true cell-cell communication edges are rarely available, we evaluate
whether inferred interactions follow these distance-scale expectations.

INPUT
-----
A single CSV file: edges_long.csv

Path (in this demo):
  /Users/taochenyang/Downloads/demo_cci_data/demo2_single_cell/edges_long.csv

Format (long table: one row = one inferred cell-cell edge):
  Columns (required):
    - dataset_id     : str   (dataset identifier; here typically 1 dataset)
    - sample_id      : str   (sample / ROI identifier; here 3 samples)
    - method         : str   (CCC method name; here 3 methods)
    - sender_id      : str   (sender cell ID)
    - receiver_id    : str   (receiver cell ID)
    - score          : float (raw interaction strength produced by the method)
    - distance       : float (Euclidean distance between sender and receiver)
    - mechanism      : str   ("contact" or "secretion")
    - sender_type    : str   (optional label; not used in metrics below)
    - receiver_type  : str   (optional label; not used in metrics below)

Typical demo size:
  - 1 dataset × 3 samples × 3 methods × (20 senders × 20 receivers) = 3600 rows
So the input table shape is approximately:
  (E, 10) where E ≈ 3600 in this demo.

INTERNAL NORMALIZATION
----------------------
We compute score_norm to make different methods’ score scales comparable:
  - Within each (dataset_id, sample_id, method, mechanism) group,
    score_norm is computed by rank-normalizing score:
      score_norm = rank(score) / n_edges_in_group
  - score_norm is in (0, 1], higher = stronger within that group.

OUTPUT (METRICS TABLES)
-----------------------
All outputs are saved to:
  /Users/taochenyang/Downloads/demo_cci_data/demo2_single_cell/

(1) metric1_distance_separation.csv   shape: (#datasets * #samples * #methods, 9)
    Per (dataset_id, sample_id, method):
      - delta_median: median(dist_secretion) - median(dist_contact)
      - ks_stat / ks_pvalue: KS test between contact vs secretion distance distributions
      - auc_contact_vs_secretion: AUROC classifying contact(1) vs secretion(0) using (-distance)
      - n_contact / n_secretion: edge counts

(2) metric2_decay_length.csv          shape: (#datasets * #samples * #methods, ~7)
    Per (dataset_id, sample_id, method):
      - lambda_contact / lambda_secretion: exponential decay length scale (fitted)
      - r2_contact / r2_secretion: fit goodness
      - lambda_ratio = lambda_secretion / lambda_contact

(3) metric2_decay_curve.parquet       shape: (~#datasets * #samples * #methods * 2 * #bins, 7)
    Binned curve points for plotting decay curves:
      - distance_bin_center: mean distance within a bin
      - mean_score_norm: mean score_norm within that bin
      - n_edges: edges per bin

(4) metric3_contact_adjacency_enrichment.csv  shape: (#datasets * #samples * #methods, 11)
    Per (dataset_id, sample_id, method), for mechanism=contact only:
      - Define adjacency threshold per sample:
          is_adjacent = 1 if distance <= (ADJ_QUANTILE)-quantile of distances in that sample
      - Define "high-score" contact edges:
          high_score = 1 if score_norm >= (HIGH_SCORE_Q)-quantile of contact edges
      - Compute enrichment of high-score edges among adjacent pairs:
          odds_ratio + 95% CI (Woolf; with Haldane-Anscombe correction if zeros)
          fisher_pvalue (one-sided, alternative="greater")
      - Also stores 2x2 counts (a,b,c,d) for debugging.

OUTPUT (FIGURES)
----------------
All figures are saved as PNGs under OUTDIR:

Metric 1 figures:
  - fig_metric1_delta_median.png
      Boxplot across samples for each method:
        y = median(dist_secretion) - median(dist_contact)
      Larger is better (stronger distance separation).

  - fig_metric1_auc.png
      Boxplot across samples for each method:
        y = AUROC(contact vs secretion using -distance)
      Higher is better (distance better separates mechanisms).

  - fig_metric1_ecdf_contact.png
      ECDF curves of distance for CONTACT edges only.
      One curve per method:
        x = distance, y = P(D <= x)
      Curves that rise earlier (more left/up) indicate more short-range contact edges.

  - fig_metric1_ecdf_secretion.png
      ECDF curves of distance for SECRETION edges only (one curve per method).

Metric 2 figures:
  - fig_metric2_lambda_ratio.png
      Boxplot across samples for each method:
        y = lambda_secretion / lambda_contact
      Typically >1 is expected (secretion longer range).

  - fig_metric2_decay_all_methods_contact.png
      Distance decay curves for CONTACT edges only (all methods in one plot).
      Built from binned averages:
        x = distance_bin_center, y = mean(score_norm)
      A “better” contact behavior usually shows higher strength at short distances and
      a clear decline as distance increases.

  - fig_metric2_decay_all_methods_secretion.png
      Same as above but for SECRETION edges only.

Metric 3 figure:
  - fig_metric3_forest_or.png
      Forest-style plot of odds ratios (with 95% CI) for contact adjacency enrichment.
      x-axis is log scale, reference line at OR=1.
      OR > 1 indicates high-score contact edges are enriched among adjacent pairs.

NOTES
-----
- N_BINS controls how many quantile-based distance bins are used to build the decay curves.
- This pipeline assumes each method produces a per-edge score and a mechanism label
  (contact vs secretion). For real benchmarks, you would first map each tool’s native
  output into this unified edges_long format.
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path

# ============================
# Paths (ALL outputs saved here)
# ============================
OUTDIR = Path("/Users/taochenyang/Downloads/demo_cci_data/demo2_single_cell")
INPATH = OUTDIR / "edges_long.csv"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
ADJ_QUANTILE = 0.10      # adjacency threshold per sample = 10% smallest distances
HIGH_SCORE_Q = 0.95      # top 5% score_norm among contact edges
N_BINS = 8              # for decay curve bins
SAVE_DPI = 200

# ---------- Helpers ----------
def rank_norm(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x, method="average")
    return r / (len(x) if len(x) > 0 else 1)

def auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUROC without sklearn:
    use Mann–Whitney U equivalence.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(y_score)  # increasing ranks
    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))

def exp_decay(d, a, lam):
    return a * np.exp(-d / lam)

def fit_decay(distance: np.ndarray, score_norm: np.ndarray):
    """
    Fit mean_score vs distance using exponential decay.
    Returns (lambda, r2). If fit fails -> (nan, nan).
    """
    if len(distance) < 10:
        return np.nan, np.nan

    qs = np.linspace(0, 1, N_BINS + 1)
    bin_edges = np.unique(np.quantile(distance, qs))
    if len(bin_edges) < 5:
        return np.nan, np.nan

    bin_ids = np.digitize(distance, bin_edges[1:-1], right=True)
    centers, means = [], []
    for b in np.unique(bin_ids):
        mask = bin_ids == b
        if mask.sum() < 5:
            continue
        centers.append(distance[mask].mean())
        means.append(score_norm[mask].mean())

    centers = np.array(centers, dtype=float)
    means = np.array(means, dtype=float)
    if len(centers) < 5:
        return np.nan, np.nan

    p0 = [max(means), np.median(distance)]
    bounds = ([0, 1e-6], [np.inf, np.inf])

    try:
        popt, _ = curve_fit(exp_decay, centers, means, p0=p0, bounds=bounds, maxfev=10000)
        a_hat, lam_hat = popt
        pred = exp_decay(centers, a_hat, lam_hat)
        ss_res = np.sum((means - pred) ** 2)
        ss_tot = np.sum((means - np.mean(means)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(lam_hat), float(r2)
    except Exception:
        return np.nan, np.nan

def or_ci_woolf(a, b, c, d, alpha=0.05):
    """
    Woolf log(OR) CI with Haldane-Anscombe correction if needed.
    """
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = stats.norm.ppf(1 - alpha/2)
    log_or = np.log(or_val)
    ci_low = np.exp(log_or - z * se)
    ci_high = np.exp(log_or + z * se)
    return float(or_val), float(ci_low), float(ci_high)

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

def ecdf_xy(x: np.ndarray):
    """Return sorted x and ECDF y."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys

# ---------- Load ----------
df = pd.read_csv(INPATH)
print("Loaded:", INPATH, " shape=", df.shape)

required_cols = [
    "dataset_id","sample_id","method","sender_id","receiver_id",
    "score","distance","mechanism","sender_type","receiver_type"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ---------- Add score_norm ----------
df["score_norm"] = df.groupby(["dataset_id","sample_id","method","mechanism"])["score"].transform(
    lambda s: rank_norm(s.to_numpy())
)

# ---------- Define adjacency ----------
adj_thr = df.groupby(["dataset_id","sample_id"])["distance"].quantile(ADJ_QUANTILE).rename("adj_thr")
df = df.merge(adj_thr.reset_index(), on=["dataset_id","sample_id"], how="left")
df["is_adjacent"] = (df["distance"] <= df["adj_thr"]).astype(int)

# =========================================================
# Metric 1: Distance separation (numbers)
# =========================================================
m1_rows = []
for (ds, sp, method), sub in df.groupby(["dataset_id","sample_id","method"]):
    Dc = sub.loc[sub["mechanism"]=="contact", "distance"].to_numpy()
    Ds = sub.loc[sub["mechanism"]=="secretion", "distance"].to_numpy()

    if len(Dc) < 5 or len(Ds) < 5:
        delta_median = np.nan
        ks_stat = np.nan
        ks_p = np.nan
        auc = np.nan
    else:
        delta_median = float(np.median(Ds) - np.median(Dc))
        ks_stat, ks_p = stats.ks_2samp(Dc, Ds, alternative="two-sided", mode="auto")
        y = np.r_[np.ones(len(Dc)), np.zeros(len(Ds))]
        s = np.r_[-Dc, -Ds]
        auc = auc_from_scores(y, s)

    m1_rows.append({
        "dataset_id": ds,
        "sample_id": sp,
        "method": method,
        "delta_median": delta_median,
        "ks_stat": float(ks_stat) if ks_stat==ks_stat else np.nan,
        "ks_pvalue": float(ks_p) if ks_p==ks_p else np.nan,
        "auc_contact_vs_secretion": auc,
        "n_contact": int(len(Dc)),
        "n_secretion": int(len(Ds)),
    })

m1 = pd.DataFrame(m1_rows)
m1.to_csv(OUTDIR / "metric1_distance_separation.csv", index=False)

# ---- Plot Metric 1: delta_median boxplot (UNCHANGED) ----
plt.figure()
methods = sorted(m1["method"].unique())
data = [m1.loc[m1["method"]==m, "delta_median"].dropna().to_numpy() for m in methods]
plt.boxplot(data, labels=methods)
plt.title("Metric 1: Distance separation (delta_median)")
plt.ylabel("median(dist_secretion) - median(dist_contact)")
plt.xlabel("method")
savefig(OUTDIR / "fig_metric1_delta_median.png")

# ---- Plot Metric 1: AUC boxplot (UNCHANGED) ----
plt.figure()
data = [m1.loc[m1["method"]==m, "auc_contact_vs_secretion"].dropna().to_numpy() for m in methods]
plt.boxplot(data, labels=methods)
plt.title("Metric 1: Distance separation (AUC)")
plt.ylabel("AUROC (contact vs secretion using -distance)")
plt.xlabel("method")
savefig(OUTDIR / "fig_metric1_auc.png")

# ---- NEW Plot Metric 1: ECDF curves (contact) ----
plt.figure()
for method in sorted(df["method"].unique()):
    sub = df[(df["mechanism"] == "contact") & (df["method"] == method)]
    xs, ys = ecdf_xy(sub["distance"].to_numpy())
    if len(xs) == 0:
        continue
    plt.plot(xs, ys, linestyle="-", marker=None, label=method)

plt.title("Metric 1 ECDF: Distance distribution (mechanism=contact)")
plt.xlabel("distance")
plt.ylabel("ECDF")
plt.legend()
savefig(OUTDIR / "fig_metric1_ecdf_contact.png")

# ---- NEW Plot Metric 1: ECDF curves (secretion) ----
plt.figure()
for method in sorted(df["method"].unique()):
    sub = df[(df["mechanism"] == "secretion") & (df["method"] == method)]
    xs, ys = ecdf_xy(sub["distance"].to_numpy())
    if len(xs) == 0:
        continue
    plt.plot(xs, ys, linestyle="-", marker=None, label=method)

plt.title("Metric 1 ECDF: Distance distribution (mechanism=secretion)")
plt.xlabel("distance")
plt.ylabel("ECDF")
plt.legend()
savefig(OUTDIR / "fig_metric1_ecdf_secretion.png")

# =========================================================
# Metric 2: Decay length scale + decay curves
# =========================================================
m2_rows = []
curve_rows = []

for (ds, sp, method, mech), sub in df.groupby(["dataset_id","sample_id","method","mechanism"]):
    dist = sub["distance"].to_numpy()
    sn = sub["score_norm"].to_numpy()

    lam, r2 = fit_decay(dist, sn)
    m2_rows.append({
        "dataset_id": ds,
        "sample_id": sp,
        "method": method,
        "mechanism": mech,
        "lambda": lam,
        "r2": r2,
    })

    # binned curve points for plotting
    if len(dist) >= 10:
        qs = np.linspace(0, 1, N_BINS + 1)
        bin_edges = np.unique(np.quantile(dist, qs))
        if len(bin_edges) >= 5:
            bin_ids = np.digitize(dist, bin_edges[1:-1], right=True)
            for b in np.unique(bin_ids):
                mask = bin_ids == b
                if mask.sum() < 5:
                    continue
                curve_rows.append({
                    "dataset_id": ds,
                    "sample_id": sp,
                    "method": method,
                    "mechanism": mech,
                    "distance_bin_center": float(dist[mask].mean()),
                    "mean_score_norm": float(sn[mask].mean()),
                    "n_edges": int(mask.sum()),
                })

m2_long = pd.DataFrame(m2_rows)

# Wide table per method/sample
m2_wide = (
    m2_long.pivot_table(
        index=["dataset_id","sample_id","method"],
        columns="mechanism",
        values=["lambda","r2"],
        aggfunc="mean"
    )
    .reset_index()
)
m2_wide.columns = [
    "_".join([c for c in col if c]) if isinstance(col, tuple) else col
    for col in m2_wide.columns
]
m2_wide["lambda_ratio"] = m2_wide["lambda_secretion"] / m2_wide["lambda_contact"]
m2_wide.to_csv(OUTDIR / "metric2_decay_length.csv", index=False)

curve_df = pd.DataFrame(curve_rows)
curve_df.to_parquet(OUTDIR / "metric2_decay_curve.parquet", index=False)

# ---- Plot Metric 2: lambda_ratio boxplot (unchanged) ----
plt.figure()
methods2 = sorted(m2_wide["method"].unique())
data = [m2_wide.loc[m2_wide["method"]==m, "lambda_ratio"].dropna().to_numpy() for m in methods2]
plt.boxplot(data, labels=methods2)
plt.title("Metric 2: Decay length ratio (lambda_secretion / lambda_contact)")
plt.ylabel("lambda_ratio")
plt.xlabel("method")
savefig(OUTDIR / "fig_metric2_lambda_ratio.png")

# ---- NEW Plot Metric 2 decay curves: ALL methods in ONE figure, split by mechanism ----
if not curve_df.empty:
    curve_plot = curve_df.copy()
    curve_plot["d_round"] = curve_plot["distance_bin_center"].round(3)

    # Aggregate across samples: mean of mean_score_norm for each (method, mechanism, d_round)
    agg = (curve_plot
           .groupby(["method","mechanism","d_round"])["mean_score_norm"]
           .mean()
           .reset_index()
           .sort_values(["mechanism","method","d_round"]))

    # Contact figure
    plt.figure()
    for method in sorted(agg["method"].unique()):
        sub = agg[(agg["mechanism"]=="contact") & (agg["method"]==method)]
        if len(sub) == 0:
            continue
        plt.plot(sub["d_round"].to_numpy(),
                 sub["mean_score_norm"].to_numpy(),
                 marker="o", linestyle="-", label=method)
    plt.title("Metric 2: Decay curves (mechanism=contact) - all methods")
    plt.xlabel("distance (binned/rounded)")
    plt.ylabel("mean score_norm")
    plt.legend()
    savefig(OUTDIR / "fig_metric2_decay_all_methods_contact.png")

    # Secretion figure
    plt.figure()
    for method in sorted(agg["method"].unique()):
        sub = agg[(agg["mechanism"]=="secretion") & (agg["method"]==method)]
        if len(sub) == 0:
            continue
        plt.plot(sub["d_round"].to_numpy(),
                 sub["mean_score_norm"].to_numpy(),
                 marker="o", linestyle="-", label=method)
    plt.title("Metric 2: Decay curves (mechanism=secretion) - all methods")
    plt.xlabel("distance (binned/rounded)")
    plt.ylabel("mean score_norm")
    plt.legend()
    savefig(OUTDIR / "fig_metric2_decay_all_methods_secretion.png")

# =========================================================
# Metric 3: Contact adjacency enrichment
# =========================================================
m3_rows = []
for (ds, sp, method), sub in df.groupby(["dataset_id","sample_id","method"]):
    sub_c = sub[sub["mechanism"]=="contact"].copy()
    if len(sub_c) < 20:
        continue

    thr = sub_c["score_norm"].quantile(HIGH_SCORE_Q)
    sub_c["high_score"] = (sub_c["score_norm"] >= thr).astype(int)

    a = int(((sub_c["high_score"]==1) & (sub_c["is_adjacent"]==1)).sum())
    b = int(((sub_c["high_score"]==1) & (sub_c["is_adjacent"]==0)).sum())
    c = int(((sub_c["high_score"]==0) & (sub_c["is_adjacent"]==1)).sum())
    d = int(((sub_c["high_score"]==0) & (sub_c["is_adjacent"]==0)).sum())

    _, pval = stats.fisher_exact([[a,b],[c,d]], alternative="greater")
    or_val, ci_low, ci_high = or_ci_woolf(a,b,c,d)

    m3_rows.append({
        "dataset_id": ds,
        "sample_id": sp,
        "method": method,
        "odds_ratio": float(or_val),
        "log2_enrichment": float(np.log2(or_val)) if or_val > 0 else np.nan,
        "fisher_pvalue": float(pval),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "a_high_adj": a, "b_high_nonadj": b, "c_low_adj": c, "d_low_nonadj": d
    })

m3 = pd.DataFrame(m3_rows)
m3.to_csv(OUTDIR / "metric3_contact_adjacency_enrichment.csv", index=False)

# ---- Plot Metric 3: forest-like OR with CI (unchanged) ----
plt.figure()
m3_sorted = m3.sort_values(["method","sample_id"]).copy()
methods3 = sorted(m3_sorted["method"].unique())

y_positions, x, xerr_low, xerr_high = [], [], [], []
for mi, m in enumerate(methods3):
    sub = m3_sorted[m3_sorted["method"]==m]
    for si, (_, row) in enumerate(sub.iterrows()):
        y = mi + (si - (len(sub)-1)/2) * 0.12
        y_positions.append(y)
        x.append(row["odds_ratio"])
        xerr_low.append(row["odds_ratio"] - row["ci_low"])
        xerr_high.append(row["ci_high"] - row["odds_ratio"])

plt.errorbar(x, y_positions, xerr=[xerr_low, xerr_high], fmt='o', capsize=3)
plt.yticks(range(len(methods3)), methods3)
plt.xscale("log")
plt.axvline(1.0, linestyle="--")
plt.title("Metric 3: Contact adjacency enrichment (OR with 95% CI)")
plt.xlabel("Odds Ratio (log scale)")
savefig(OUTDIR / "fig_metric3_forest_or.png")

print("\nAll outputs saved to:", OUTDIR)
print("Tables:")
print(" - metric1_distance_separation.csv")
print(" - metric2_decay_length.csv")
print(" - metric2_decay_curve.parquet")
print(" - metric3_contact_adjacency_enrichment.csv")
print("Figures:")
print(" - fig_metric1_delta_median.png")
print(" - fig_metric1_auc.png")
print(" - fig_metric1_ecdf_contact.png")
print(" - fig_metric1_ecdf_secretion.png")
print(" - fig_metric2_lambda_ratio.png")
print(" - fig_metric2_decay_all_methods_contact.png")
print(" - fig_metric2_decay_all_methods_secretion.png")
print(" - fig_metric3_forest_or.png")
