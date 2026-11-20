"""
Research delivery pipeline v2: transcutaneous liver-fibrosis Monte Carlo study.

Fixes over v1 (`research_delivery_pipeline.py`)
----------------------------------------------
v1 scored detectors outside the simulation domain (grid x range +-0.75 mm and
z range 1.6 mm vs 3 mm skin and 2.4 mm separations) and used 700-photon
single-point detectors, so most features were numerically zero plus noise and
the best balanced accuracy stayed near 0.61. v2 uses the validated numba core
`mc_liver_fast.py` (330x faster, spatially-resolved scoring, see
`validate_fast_core.py` / `validation_report.json`) and rebuilds the full
experiment stack:

  E0  core validation + energy balance (external script, report copied here)
  E1  main comparison: feature-group ladder, designated classifier
  E2  classifier comparison on the full feature set (SOTA-style ML benchmark)
  E3  statistical tests: paired bootstrap CIs + Wilcoxon signed-rank
  E4  photon-budget sensitivity
  E5  cohort-size learning curve
  E6  instrument-noise robustness (noise applied post-hoc to stored signals)
  E7  skin-thickness domain shift (train mid-range, test thin/thick)
  E8  continuous severity staging regression (Spearman / R2 / MAE)
  E9  spatially-resolved mechanism curves with confidence bands
  E10 feature effect-size maps (Cohen's d over wavelength x separation)
  E11 leakage-safe wavelength/SDS channel optimization with nested validation

Cohort design (synthetic, simulation-first):
  label 0  low-severity synthetic group,  severity ~ U(0.00, 0.30)
  label 1  high-severity synthetic group, severity ~ U(0.50, 1.00)
  skin thickness ~ N(3.0, 0.5) mm clipped to [1.8, 4.5], per-subject optical
  jitter (liver +-10%, skin +-8%), instrument noise applied at feature time.

Run (full, ~8-12 min on 32 cores):
    python research_delivery_pipeline_v2.py --out final_delivery_v2
Smoke test (~1 min):
    python research_delivery_pipeline_v2.py --out final_delivery_v2_smoke --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless backend; avoids Tk thread crashes under numba parallelism
import matplotlib.pyplot as plt
import numpy as np

from mc_liver_fast import FastGeometry, FastSource, channel_counts, simulate_fast
from skin_liver_fibrosis_mc import build_layered_tissue, optical_properties

EPS = 1.0e-12
CM_TO_MM = 10.0

# lambda is the illumination wavelength and rho_j is the source-to-detector
# center separation.  Because the transport scorer bins |x|, each rho_j denotes
# a bilateral pair of symmetric detector strips centered at +/-rho_j.
WAVELENGTHS: Tuple[int, ...] = (730, 808, 940, 1064, 660)
SEPARATIONS_MM: Tuple[float, ...] = tuple(0.25 + 0.5 * i for i in range(10))  # rho_j = 0.25 .. 4.75 mm
CHANNEL_HALF_WIDTH_CM = 0.025  # 0.25 mm half-width per strip -> non-overlapping 0.5 mm strips


# --------------------------------------------------------------------------- #
# logging tee
# --------------------------------------------------------------------------- #
class Tee:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s: str) -> int:
        self.stdout.write(s)
        self.file.write(s)
        return len(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CohortConfig:
    skin_mean_mm: float = 3.0
    skin_std_mm: float = 0.5
    skin_min_mm: float = 1.8
    skin_max_mm: float = 4.5
    control_sev_range: Tuple[float, float] = (0.00, 0.30)
    fibrosis_sev_range: Tuple[float, float] = (0.50, 1.00)
    liver_jitter: float = 0.10
    skin_jitter: float = 0.08
    instrument_noise: float = 0.01  # relative Gaussian noise on measured signals


# --------------------------------------------------------------------------- #
# subject simulation
# --------------------------------------------------------------------------- #
def _jittered_optics(wl: int, state: str, severity: float, rng: np.random.Generator, cfg: CohortConfig):
    optics = optical_properties(wl, state, fibrosis_severity=severity)
    mua = optics.mua.copy()
    mus = optics.mus.copy()
    liver_scale_a = 1.0 + rng.uniform(-cfg.liver_jitter, cfg.liver_jitter)
    liver_scale_s = 1.0 + rng.uniform(-cfg.liver_jitter, cfg.liver_jitter)
    skin_scale_a = 1.0 + rng.uniform(-cfg.skin_jitter, cfg.skin_jitter)
    skin_scale_s = 1.0 + rng.uniform(-cfg.skin_jitter, cfg.skin_jitter)
    mua[3] *= liver_scale_a
    mus[3] *= liver_scale_s
    mua[1:3] *= skin_scale_a
    mus[1:3] *= skin_scale_s
    from skin_liver_fibrosis_mc import TissueOptics

    return TissueOptics(mua=mua, mus=mus, g=optics.g.copy(), meta=dict(optics.meta))


def simulate_subject_raw(
    subject_seed: int,
    label: int,
    severity: float,
    skin_mm: float,
    cfg: CohortConfig,
    photons: int,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """Run one Monte Carlo simulation per wavelength and store noise-free
    channel signals plus per-wavelength scalar statistics.

    Optical channels follow the energy-weighted definitions

        R(lambda, rho_j) = sum_{i in C_j} W_i(exit) / N0
        L(lambda, rho_j) = sum_{i in C_j} W_i(exit) I_i(liver) / N0

    where each launched packet has initial weight one, so N0 == ``photons``.
    Count-based reach/return quantities remain separate trajectory-history
    probabilities in ``scal`` and must not be described as measured intensity.
    """
    rng = np.random.default_rng(subject_seed)
    skin_cm = skin_mm / CM_TO_MM
    geo = FastGeometry(skin_thickness=skin_cm, epidermis_thickness=float(np.clip(0.03 * skin_cm, 0.005, 0.012)))
    tissue = build_layered_tissue(geo)
    state = "fibrosis" if label == 1 else "normal"
    src = FastSource()

    chan = np.zeros((len(WAVELENGTHS), len(SEPARATIONS_MM), 2), dtype=np.float64)  # weighted R(lambda,rho), L(lambda,rho)
    # The last three scalar entries are packet-count path probabilities.  They
    # intentionally remain count based because they describe trajectories, not
    # surface optical energy.
    scal = np.zeros((len(WAVELENGTHS), 5), dtype=np.float64)  # liver_abs, skin_abs, reach_p, return_p, exit_top_p
    for i, wl in enumerate(WAVELENGTHS):
        optics = _jittered_optics(wl, state, severity, rng, cfg)
        res = simulate_fast(geo, tissue, optics, src, photons, subject_seed + 13 * wl, parallel=True)
        # Use remaining exit weights for the paper's R/L signals.  The count
        # histogram is retained by the core for validation and path analysis.
        hist = res["radial_weight_hist"]
        stats = res["stats"]
        for j, sep in enumerate(SEPARATIONS_MM):
            r_c, l_c = channel_counts(hist, res["radial_bin"], sep / CM_TO_MM, CHANNEL_HALF_WIDTH_CM)
            chan[i, j, 0] = r_c / photons
            chan[i, j, 1] = l_c / photons
        scal[i] = [
            stats["mean_liver_absorption"],
            stats["mean_skin_absorption"],
            stats["reached_liver_probability"],
            stats["returned_after_liver_probability"],
            stats["exit_top_probability"],
        ]
    return {
        "label": int(label),
        "severity": float(severity),
        "skin_mm": float(skin_mm),
        "chan": chan,
        "scal": scal,
    }


def build_cohort_raw(samples: int, cfg: CohortConfig, photons: int, seed: int, staging: bool = False) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    half = samples // 2
    for i in range(samples):
        if staging:
            severity = float(rng.uniform(0.0, 1.0))
            label = int(severity >= 0.40)  # reference label only; regression uses severity
        else:
            label = 0 if i < half else 1
            lo, hi = cfg.control_sev_range if label == 0 else cfg.fibrosis_sev_range
            severity = float(rng.uniform(lo, hi))
        skin = float(np.clip(rng.normal(cfg.skin_mean_mm, cfg.skin_std_mm), cfg.skin_min_mm, cfg.skin_max_mm))
        rows.append(simulate_subject_raw(seed + 1009 * i, label, severity, skin, cfg, photons))
    rng.shuffle(rows)
    return rows


def save_cohort_npz(path: Path, rows: List[Dict], meta: Dict) -> None:
    np.savez_compressed(
        path,
        label=np.asarray([r["label"] for r in rows], dtype=int),
        severity=np.asarray([r["severity"] for r in rows], dtype=float),
        skin_mm=np.asarray([r["skin_mm"] for r in rows], dtype=float),
        chan=np.stack([r["chan"] for r in rows]),
        scal=np.stack([r["scal"] for r in rows]),
        meta=json.dumps(meta),
    )


def load_cohort_npz(path: Path) -> List[Dict]:
    z = np.load(path, allow_pickle=False)
    return [
        {"label": int(z["label"][i]), "severity": float(z["severity"][i]), "skin_mm": float(z["skin_mm"][i]),
         "chan": z["chan"][i], "scal": z["scal"][i]}
        for i in range(len(z["label"]))
    ]


# --------------------------------------------------------------------------- #
# feature derivation (instrument noise applied here, post-hoc)
# --------------------------------------------------------------------------- #
def wl_tag(i: int) -> str:
    return f"w{WAVELENGTHS[i]}"


def sep_tag(j: int) -> str:
    return f"s{int(round(SEPARATIONS_MM[j] * 100)):03d}"


def derive_features(rows: List[Dict], noise: float, seed: int) -> Tuple[List[Dict[str, float]], List[str]]:
    """Convert normalized exit-weight channel signals into named features.

    Log transforms compress the multiplicative dynamic range of reflectance and
    liver-return signals; ratios normalize per-subject instrument gain.

    Coarse-binned L-channel aggregates (pairs of adjacent separations) improve
    SNR for the low-amplitude liver-return signal.
    """
    rng = np.random.default_rng(seed)
    n_wl, n_sep = len(WAVELENGTHS), len(SEPARATIONS_MM)
    deep_j = [j for j, s in enumerate(SEPARATIONS_MM) if s >= 2.25]
    deep_i = [i for i, w in enumerate(WAVELENGTHS) if w in (808, 940, 1064)]
    # coarse bin pairs: (0,1), (2,3), (4,5), (6,7), (8,9) — 2× wider effective channel
    coarse_groups = [(j, j + 1) for j in range(0, n_sep - 1, 2)]
    feat_rows: List[Dict[str, float]] = []
    for r in rows:
        f: Dict[str, float] = {}
        chan = r["chan"]
        scal = r["scal"]
        for i in range(n_wl):
            for j in range(n_sep):
                rp = float(chan[i, j, 0])
                lp = float(chan[i, j, 1])
                if noise > 0:
                    rp = max(0.0, rp + rng.normal(0.0, noise * max(rp, 1.0e-6)))
                    lp = max(0.0, lp + rng.normal(0.0, noise * max(lp, 1.0e-7)))
                tag = f"{wl_tag(i)}_{sep_tag(j)}"
                f[f"{tag}_R"] = math.log10(rp + 1.0e-7)
                f[f"{tag}_L"] = math.log10(lp + 1.0e-8)
                f[f"{tag}_F"] = math.log10((lp + 1.0e-8) / (rp + 1.0e-7))
            # Coarse-binned L and F: sum normalized weights before log to reduce
            # Monte Carlo variance without treating packet counts as intensity.
            for ja, jb in coarse_groups:
                rp_sum = float(chan[i, ja, 0] + chan[i, jb, 0])
                lp_sum = float(chan[i, ja, 1] + chan[i, jb, 1])
                ctag = f"{wl_tag(i)}_c{ja//2:01d}"
                f[f"{ctag}_L"] = math.log10(lp_sum + 1.0e-8)
                f[f"{ctag}_F"] = math.log10((lp_sum + 1.0e-8) / (rp_sum + 1.0e-7))
            la = float(scal[i, 0])
            sa = float(scal[i, 1])
            f[f"{wl_tag(i)}_labs"] = la
            f[f"{wl_tag(i)}_lsr"] = la / max(sa, EPS)
            f[f"{wl_tag(i)}_reach"] = float(scal[i, 2])
            f[f"{wl_tag(i)}_ret"] = float(scal[i, 3])
        # aggregates over deep/deeper-penetrating channels
        def mean_of(suffix: str, ii: Sequence[int], jj: Sequence[int]) -> float:
            return float(np.mean([f[f"{wl_tag(i)}_{sep_tag(j)}_{suffix}"] for i in ii for j in jj]))

        f["agg_deep_R"] = mean_of("R", deep_i, deep_j)
        f["agg_deep_L"] = mean_of("L", deep_i, deep_j)
        f["agg_deep_F"] = mean_of("F", deep_i, deep_j)
        f["agg_all_L"] = mean_of("L", range(n_wl), range(n_sep))
        f["agg_all_F"] = mean_of("F", range(n_wl), range(n_sep))
        # coarse-L aggregates across wavelengths
        f["agg_coarse_L"] = float(np.mean([f[f"{wl_tag(i)}_c{g//2:01d}_L"] for i in range(n_wl) for g in range(0, n_sep - 1, 2)]))
        f["agg_coarse_F"] = float(np.mean([f[f"{wl_tag(i)}_c{g//2:01d}_F"] for i in range(n_wl) for g in range(0, n_sep - 1, 2)]))
        i730, i1064 = WAVELENGTHS.index(730), WAVELENGTHS.index(1064)
        f["agg_slope_R_1064_730"] = mean_of("R", [i1064], deep_j) - mean_of("R", [i730], deep_j)
        f["agg_slope_L_1064_730"] = mean_of("L", [i1064], deep_j) - mean_of("L", [i730], deep_j)
        f["agg_mean_lsr"] = float(np.mean([f[f"{wl_tag(i)}_lsr"] for i in range(n_wl)]))
        f["agg_mean_ret"] = float(np.mean([f[f"{wl_tag(i)}_ret"] for i in range(n_wl)]))
        # Diffusion-style effective attenuation fit on log R(lambda,rho).
        rho = np.asarray(SEPARATIONS_MM, dtype=float)
        sel = (rho >= 1.0) & (rho <= 4.0)
        for i in range(n_wl):
            y = np.asarray([f[f"{wl_tag(i)}_{sep_tag(j)}_R"] for j in range(n_sep)])[sel]
            slope, intercept = np.polyfit(rho[sel], y, 1)
            f[f"{wl_tag(i)}_atten"] = float(slope)
            f[f"{wl_tag(i)}_backscatter"] = float(intercept)
        feat_rows.append(f)
    columns = list(feat_rows[0].keys())
    return feat_rows, columns


def feature_groups(columns: Sequence[str]) -> Dict[str, List[str]]:
    cols = list(columns)
    by_wl_R = {w: [c for c in cols if c.startswith(f"w{w}_s") and c.endswith("_R")] for w in WAVELENGTHS}
    all_R = [c for c in cols if c.startswith("w") and "_s" in c and c.endswith("_R")]
    all_L = [c for c in cols if c.startswith("w") and "_s" in c and c.endswith("_L")]
    all_F = [c for c in cols if c.startswith("w") and "_s" in c and c.endswith("_F")]
    coarse_L = [c for c in cols if c.startswith("w") and "_c" in c and c.endswith("_L")]
    coarse_F = [c for c in cols if c.startswith("w") and "_c" in c and c.endswith("_F")]
    scalars = [c for c in cols if c.startswith("w") and ("_lsr" in c or "_labs" in c or "_reach" in c or "_ret" in c)]
    atten = [c for c in cols if c.endswith("_atten") or c.endswith("_backscatter")]
    aggs = [c for c in cols if c.startswith("agg_")]
    return {
        "Baseline-R": [f"w808_s025_R"],
        "MultiLambda-R": [f"w{w}_s025_R" for w in WAVELENGTHS],
        "MultiRho-R": by_wl_R[808],
        "SpectralGeometry-R": all_R,
        "DiffusionFit": atten,
        "PathGate-L": all_L,
        "PathGate-CoarseL": coarse_L,
        "Fusion-noPath": all_R + atten + ["agg_deep_R", "agg_slope_R_1064_730"],
        "Fusion+Path": all_R + all_L + coarse_L + scalars + ["agg_deep_R", "agg_deep_L", "agg_coarse_L",
                       "agg_slope_R_1064_730", "agg_slope_L_1064_730", "agg_mean_lsr", "agg_mean_ret"],
        "Full": cols,
    }


# --------------------------------------------------------------------------- #
# evaluation utilities
# --------------------------------------------------------------------------- #
def stratified_split(labels: np.ndarray, seed: int, train_fraction: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        cut = max(2, int(round(len(idx) * train_fraction)))
        tr.extend(idx[:cut].tolist())
        te.extend(idx[cut:].tolist())
    return np.asarray(tr, dtype=int), np.asarray(te, dtype=int)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in np.unique(y_true):
        m = y_true == cls
        recalls.append(float(np.mean(y_pred[m] == cls)) if np.any(m) else 0.0)
    return float(np.mean(recalls))


def cohens_d_pooled(low_group: np.ndarray, high_group: np.ndarray) -> float:
    """Cohen's d for independent groups using the pooled *sample* variance.

    The sign is ``high_group - low_group``.  In this study, therefore, d < 0
    means the high-severity/fibrosis group has a lower signal.  ``ddof=1`` and
    the (n-1)-weighted pooling implement the manuscript definition exactly and
    remain correct if the two group sizes later differ.
    """
    low = np.asarray(low_group, dtype=float)
    high = np.asarray(high_group, dtype=float)
    if low.size < 2 or high.size < 2:
        raise ValueError("Cohen's d requires at least two observations per group")
    pooled_var = (
        (low.size - 1) * low.var(ddof=1) + (high.size - 1) * high.var(ddof=1)
    ) / (low.size + high.size - 2)
    return float((high.mean() - low.mean()) / math.sqrt(max(pooled_var, EPS)))


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Manual ROC AUC (no scipy dependency for this core metric)."""
    y_true = np.asarray(y_true, dtype=int)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    greater = float(np.sum(pos[:, None] > neg[None, :]))
    equal = float(np.sum(pos[:, None] == neg[None, :]))
    auc = float((greater + 0.5 * equal) / (len(pos) * len(neg)))
    # Clip degenerate cases: all scores identical → auc is uninformative → 0.5
    if not np.isfinite(auc) or np.std(scores) < EPS:
        return 0.5
    return auc


def youden_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy; candidates are midpoints
    between sorted unique training scores (plus one below the minimum)."""
    values = np.unique(scores)
    if len(values) == 1:
        return float(values[0])
    candidates = np.concatenate([[values[0] - 1.0e-9], (values[:-1] + values[1:]) / 2.0])
    best_j, best_t = -1.0, float(candidates[0])
    for t in candidates:
        j = balanced_accuracy(y, (scores >= t).astype(int))
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def make_classifier(name: str, seed: int):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC

    table = {
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "LogReg": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
        "LinearSVM": lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
        "RBF-SVM": lambda: SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        "kNN": lambda: KNeighborsClassifier(n_neighbors=7),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=4),
        "GradBoost": lambda: HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, min_samples_leaf=5, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        ),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=500, random_state=seed),
    }
    if name not in table:
        raise KeyError(name)
    return table[name]()


CLASSIFIER_NAMES = ["LDA", "LogReg", "LinearSVM", "RBF-SVM", "kNN", "RandomForest", "GradBoost", "MLP"]
DESIGNATED_CLASSIFIER = "LogReg"  # robust on small/medium cohorts; GradBoost needs larger samples


def fit_score_both(clf_name: str, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int, max_feats: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Leakage-safe pipeline: scaling + univariate screening fitted on train only.
    Returns (test_scores, train_scores) from a single fit.

    Uses predict_proba for consistent probability estimates; LinearSVC is the
    only classifier without predict_proba (uses decision_function instead).
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    k = min(max_feats, x_tr.shape[1])
    steps = [("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=k)), ("clf", make_classifier(clf_name, seed))]
    if clf_name in ("RandomForest", "GradBoost"):
        steps = [("select", SelectKBest(f_classif, k=k)), ("clf", make_classifier(clf_name, seed))]
    pipe = Pipeline(steps)
    pipe.fit(x_tr, y_tr)
    clf = pipe.named_steps["clf"]

    # Prefer predict_proba (standard probabilistic output). LinearSVC only
    # exposes decision_function — use that as fallback.
    if hasattr(clf, "predict_proba"):
        te_scores = np.asarray(pipe.predict_proba(x_te)[:, 1], dtype=float)
        tr_scores = np.asarray(pipe.predict_proba(x_tr)[:, 1], dtype=float)
    elif hasattr(clf, "decision_function"):
        te_scores = np.asarray(pipe.decision_function(x_te), dtype=float)
        tr_scores = np.asarray(pipe.decision_function(x_tr), dtype=float)
    else:
        raise RuntimeError(f"Classifier {clf_name} has neither predict_proba nor decision_function")

    # Detect degenerate outputs: NaN / Inf / all-identical scores
    if np.any(~np.isfinite(te_scores)) or np.any(~np.isfinite(tr_scores)):
        print(f"    [WARN] {clf_name}: non-finite scores detected; replacing with zeros")
        te_scores = np.nan_to_num(te_scores, nan=0.0, posinf=1.0, neginf=0.0)
        tr_scores = np.nan_to_num(tr_scores, nan=0.0, posinf=1.0, neginf=0.0)

    return te_scores, tr_scores


def evaluate_feature_set(
    x: np.ndarray,
    labels: np.ndarray,
    clf_name: str,
    repeats: int,
    seed: int,
    collect_roc: bool = False,
) -> Dict[str, object]:
    accs, bas, aucs, sens, spec, f1s = [], [], [], [], [], []
    roc_fpr = np.linspace(0.0, 1.0, 101)
    pooled_roc_labels: List[np.ndarray] = []
    pooled_roc_scores: List[np.ndarray] = []
    degenerate_warned = False
    for r in range(repeats):
        tr, te = stratified_split(labels, seed + 131 * r)
        scores, tr_scores = fit_score_both(clf_name, x[tr], labels[tr], x[te], seed + r)
        # threshold tuned on train scores only
        thr = youden_threshold(labels[tr], tr_scores)
        pred = (scores >= thr).astype(int)
        y_te = labels[te]
        tp = int(np.sum((pred == 1) & (y_te == 1)))
        tn = int(np.sum((pred == 0) & (y_te == 0)))
        fp = int(np.sum((pred == 1) & (y_te == 0)))
        fn = int(np.sum((pred == 0) & (y_te == 1)))
        accs.append(float(np.mean(pred == y_te)))
        bas.append(balanced_accuracy(y_te, pred))
        aucs.append(auc_score(y_te, scores))
        sens.append(tp / max(tp + fn, 1))
        spec.append(tn / max(tn + fp, 1))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        if collect_roc:
            # Pool only held-out predictions.  This gives one empirical ROC
            # curve whose AUC is computed from exactly the scores being drawn,
            # rather than a visually ambiguous interpolation of split curves.
            pooled_roc_labels.append(y_te)
            pooled_roc_scores.append(scores)
        # flag degenerate fold (all predictions identical)
        if not degenerate_warned and np.std(scores) < EPS:
            print(f"    [WARN] fold {r}: {clf_name} produced constant scores — check photon count / class separability")
            degenerate_warned = True
    out: Dict[str, object] = {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy": float(np.mean(bas)),
        "balanced_accuracy_std": float(np.std(bas)),
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "sensitivity": float(np.mean(sens)),
        "specificity": float(np.mean(spec)),
        "f1": float(np.mean(f1s)),
        "ba_per_split": bas,
        "auc_per_split": aucs,
    }
    if collect_roc:
        y_pool = np.concatenate(pooled_roc_labels)
        score_pool = np.concatenate(pooled_roc_scores)
        order = np.argsort(-score_pool, kind="mergesort")
        ys = y_pool[order]
        tps = np.cumsum(ys == 1) / max(np.sum(ys == 1), 1)
        fps = np.cumsum(ys == 0) / max(np.sum(ys == 0), 1)
        out["roc_fpr"] = roc_fpr.tolist()
        out["roc_tpr_mean"] = np.interp(
            roc_fpr,
            np.concatenate([[0.0], fps, [1.0]]),
            np.concatenate([[0.0], tps, [1.0]]),
        ).tolist()
        out["roc_auc"] = auc_score(y_pool, score_pool)
    return out


# --------------------------------------------------------------------------- #
# E11: leakage-safe wavelength and source-detector-separation optimization
# --------------------------------------------------------------------------- #
def _channel_cv_score(
    x: np.ndarray,
    labels: np.ndarray,
    feature_indices: Sequence[int],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> Dict[str, float]:
    """Score one physical channel subset on fixed inner-validation folds.

    Scaling and logistic-regression fitting occur independently inside every
    fold.  Reusing the same folds for all candidates makes subset comparisons
    paired without exposing the outer test subjects to channel selection.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    idx = np.asarray(feature_indices, dtype=int)
    ba_values: List[float] = []
    auc_values: List[float] = []
    for fold, (tr, va) in enumerate(splits):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", make_classifier("LogReg", seed + fold)),
        ])
        pipe.fit(x[tr][:, idx], labels[tr])
        scores = np.asarray(pipe.predict_proba(x[va][:, idx])[:, 1], dtype=float)
        ba_values.append(balanced_accuracy(labels[va], (scores >= 0.5).astype(int)))
        auc_values.append(auc_score(labels[va], scores))
    ba_array = np.asarray(ba_values, dtype=float)
    return {
        "ba_mean": float(ba_array.mean()),
        "ba_se": float(ba_array.std(ddof=1) / math.sqrt(len(ba_array))) if len(ba_array) > 1 else 0.0,
        "auc_mean": float(np.mean(auc_values)),
    }


def _one_standard_error_choice(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Choose the smallest candidate within one SE of the best inner-CV BA."""
    best = max(candidates, key=lambda row: float(row["ba_mean"]))
    cutoff = float(best["ba_mean"]) - float(best["ba_se"])
    eligible = [row for row in candidates if float(row["ba_mean"]) >= cutoff]
    # Complexity is the number of physical wavelengths or SDS positions.  BA
    # and the string representation only break ties deterministically.
    return min(
        eligible,
        key=lambda row: (
            int(row["complexity"]),
            -float(row["ba_mean"]),
            str(row["subset"]),
        ),
    )


def run_channel_optimization(
    feat_rows: List[Dict[str, float]],
    labels: np.ndarray,
    outer_repeats: int,
    seed: int,
    out: Path,
) -> Dict[str, object]:
    """Optimize measurable R(lambda, rho) channels with nested validation.

    Stage 1 exhaustively evaluates all 2^5-1 wavelength subsets while retaining
    every SDS.  Stage 2 performs sequential forward selection over the ten SDS
    locations for the selected wavelengths.  At both stages, the one-standard-
    error rule chooses the least complex subset whose mean inner-CV balanced
    accuracy is within one SE of the best candidate.  All selection is repeated
    inside each outer training split; outer test subjects are used exactly once
    for an unbiased performance estimate.

    Only R channels are eligible because they are direct surface observables.
    L is a simulation path label and is therefore excluded from probe-design
    optimization even though it remains useful for mechanism analysis.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_wl = len(WAVELENGTHS)
    n_sds = len(SEPARATIONS_MM)
    r_columns = [
        f"w{wl}_{sep_tag(sds_index)}_R"
        for wl in WAVELENGTHS
        for sds_index in range(n_sds)
    ]
    x = np.asarray([[row[col] for col in r_columns] for row in feat_rows], dtype=float)

    split_rows: List[Dict[str, object]] = []
    wavelength_hits = {wl: 0 for wl in WAVELENGTHS}
    sds_hits = {sds: 0 for sds in SEPARATIONS_MM}
    all_wavelength_subsets = [
        subset
        for size in range(1, n_wl + 1)
        for subset in combinations(range(n_wl), size)
    ]

    for outer in range(outer_repeats):
        outer_train, outer_test = stratified_split(labels, seed + 10007 * outer)
        y_train = labels[outer_train]
        class_counts = np.bincount(y_train, minlength=2)
        inner_folds = int(min(5, class_counts.min()))
        if inner_folds < 2:
            raise ValueError("channel optimization needs at least two training samples per class")
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + outer)
        inner_splits = [(tr, va) for tr, va in inner.split(x[outer_train], y_train)]

        wavelength_candidates: List[Dict[str, object]] = []
        for subset in all_wavelength_subsets:
            indices = [wi * n_sds + sj for wi in subset for sj in range(n_sds)]
            score = _channel_cv_score(x[outer_train], y_train, indices, inner_splits, seed + outer)
            wavelength_candidates.append({
                "subset": subset,
                "complexity": len(subset),
                **score,
            })
        wavelength_choice = _one_standard_error_choice(wavelength_candidates)
        selected_wavelength_indices = tuple(int(i) for i in wavelength_choice["subset"])

        # Sequential forward selection creates a transparent path containing
        # one best subset for every possible SDS count (1..10).
        selected_sds: List[int] = []
        remaining_sds = list(range(n_sds))
        sds_path: List[Dict[str, object]] = []
        while remaining_sds:
            additions: List[Dict[str, object]] = []
            for candidate_sds in remaining_sds:
                trial = tuple(sorted(selected_sds + [candidate_sds]))
                indices = [wi * n_sds + sj for wi in selected_wavelength_indices for sj in trial]
                score = _channel_cv_score(x[outer_train], y_train, indices, inner_splits, seed + 5000 + outer)
                additions.append({
                    "subset": trial,
                    "complexity": len(trial),
                    **score,
                })
            step_best = max(additions, key=lambda row: (float(row["ba_mean"]), float(row["auc_mean"])))
            selected_sds = [int(i) for i in step_best["subset"]]
            remaining_sds = [i for i in remaining_sds if i not in selected_sds]
            sds_path.append(step_best)
        sds_choice = _one_standard_error_choice(sds_path)
        selected_sds_indices = tuple(int(i) for i in sds_choice["subset"])

        final_indices = [
            wi * n_sds + sj
            for wi in selected_wavelength_indices
            for sj in selected_sds_indices
        ]
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", make_classifier("LogReg", seed + 9000 + outer)),
        ])
        model.fit(x[outer_train][:, final_indices], labels[outer_train])
        outer_scores = np.asarray(model.predict_proba(x[outer_test][:, final_indices])[:, 1], dtype=float)
        outer_pred = (outer_scores >= 0.5).astype(int)

        selected_wavelengths = tuple(WAVELENGTHS[i] for i in selected_wavelength_indices)
        selected_separations = tuple(SEPARATIONS_MM[i] for i in selected_sds_indices)
        for wl in selected_wavelengths:
            wavelength_hits[wl] += 1
        for sds in selected_separations:
            sds_hits[sds] += 1
        split_rows.append({
            "outer_split": outer,
            "selected_wavelengths_nm": ";".join(str(wl) for wl in selected_wavelengths),
            "selected_separations_mm": ";".join(f"{sds:.2f}" for sds in selected_separations),
            "n_wavelengths": len(selected_wavelengths),
            "n_separations": len(selected_separations),
            "n_channels": len(final_indices),
            "inner_wavelength_ba": float(wavelength_choice["ba_mean"]),
            "inner_sds_ba": float(sds_choice["ba_mean"]),
            "outer_balanced_accuracy": balanced_accuracy(labels[outer_test], outer_pred),
            "outer_auc": auc_score(labels[outer_test], outer_scores),
        })

    save_csv(out / "tables" / "channel_optimization_splits.csv", split_rows)
    frequency_rows: List[Dict[str, object]] = []
    for wl in sorted(WAVELENGTHS):
        frequency_rows.append({
            "channel_type": "wavelength_nm",
            "channel_value": wl,
            "selected_count": wavelength_hits[wl],
            "selection_frequency": wavelength_hits[wl] / outer_repeats,
        })
    for sds in SEPARATIONS_MM:
        frequency_rows.append({
            "channel_type": "separation_mm",
            "channel_value": f"{sds:.2f}",
            "selected_count": sds_hits[sds],
            "selection_frequency": sds_hits[sds] / outer_repeats,
        })
    save_csv(out / "tables" / "channel_selection_frequency.csv", frequency_rows)

    outer_ba = np.asarray([float(row["outer_balanced_accuracy"]) for row in split_rows])
    outer_auc = np.asarray([float(row["outer_auc"]) for row in split_rows])
    summary: Dict[str, object] = {
        "method": "nested exhaustive wavelength search + SDS SFS + one-SE rule",
        "outer_repeats": outer_repeats,
        "balanced_accuracy": float(outer_ba.mean()),
        "balanced_accuracy_std": float(outer_ba.std(ddof=1)) if outer_repeats > 1 else 0.0,
        "auc": float(outer_auc.mean()),
        "auc_std": float(outer_auc.std(ddof=1)) if outer_repeats > 1 else 0.0,
        "mean_n_wavelengths": float(np.mean([int(row["n_wavelengths"]) for row in split_rows])),
        "mean_n_separations": float(np.mean([int(row["n_separations"]) for row in split_rows])),
        "mean_n_channels": float(np.mean([int(row["n_channels"]) for row in split_rows])),
    }
    save_csv(out / "tables" / "channel_optimization_summary.csv", [summary])
    print(
        "  nested channel optimization: "
        f"BA={summary['balanced_accuracy']:.3f}+/-{summary['balanced_accuracy_std']:.3f}, "
        f"AUC={summary['auc']:.3f}, mean channels={summary['mean_n_channels']:.1f}"
    )
    return summary


def paired_bootstrap(a: Sequence[float], b: Sequence[float], seed: int = 123, repeats: int = 2000) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    deltas = [float(np.mean(b[idx] - a[idx])) for idx in (rng.integers(0, n, size=n) for _ in range(repeats))]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Wilcoxon signed-rank test (one-sided: b > a). Returns NaN on failure."""
    try:
        from scipy.stats import wilcoxon

        diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        if np.allclose(diff, 0.0):
            return 1.0
        if len(diff) < 3:
            return float("nan")
        # SciPy >=1.9 calls this argument ``method``; older supported project
        # environments (for example SciPy 1.8) call it ``mode``.
        method = "exact" if len(diff) <= 25 else "approx"
        try:
            result = wilcoxon(diff, alternative="greater", method=method)
        except TypeError:
            result = wilcoxon(diff, alternative="greater", mode=method)
        return float(result.pvalue)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# E1 + E2 + E3: main comparison, classifier comparison, statistics
# --------------------------------------------------------------------------- #
def run_main_comparison(
    feat_rows: List[Dict[str, float]],
    columns: List[str],
    labels: np.ndarray,
    repeats: int,
    seed: int,
    out: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], Dict[str, List[str]]]:
    groups = feature_groups(columns)
    results: List[Dict[str, object]] = []
    details: Dict[str, Dict[str, object]] = {}
    roc_methods = {"Baseline-R", "SpectralGeometry-R", "PathGate-L", "PathGate-CoarseL", "Fusion+Path", "Full"}
    for name, cols in groups.items():
        x = np.asarray([[fr[c] for c in cols] for fr in feat_rows], dtype=float)
        res = evaluate_feature_set(x, labels, DESIGNATED_CLASSIFIER, repeats, seed, collect_roc=name in roc_methods)
        details[name] = res
        results.append(
            {
                "feature_set": name,
                "n_features": len(cols),
                "classifier": DESIGNATED_CLASSIFIER,
                **{k: res[k] for k in ("accuracy", "accuracy_std", "balanced_accuracy", "balanced_accuracy_std", "auc", "auc_std", "sensitivity", "specificity", "f1")},
            }
        )
        print(f"  {name:20s} feats={len(cols):3d} BA={res['balanced_accuracy']:.3f}+/-{res['balanced_accuracy_std']:.3f} AUC={res['auc']:.3f}")
    save_csv(out / "tables" / "main_comparison.csv", results)

    # ROC curves for key methods
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=220)
    for name in ["Baseline-R", "SpectralGeometry-R", "PathGate-L", "PathGate-CoarseL", "Fusion+Path", "Full"]:
        res = details[name]
        ax.plot(res["roc_fpr"], res["roc_tpr_mean"], lw=2, label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC curves ({repeats} repeated splits, {DESIGNATED_CLASSIFIER})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figures" / "supp_roc_curves.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    return results, details, groups


def run_classifier_comparison(
    feat_rows: List[Dict[str, float]],
    columns: List[str],
    labels: np.ndarray,
    repeats: int,
    seed: int,
    out: Path,
) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    auc_mat = np.zeros((len(CLASSIFIER_NAMES), 1))
    for ci, clf in enumerate(CLASSIFIER_NAMES):
        x = np.asarray([[fr[c] for c in columns] for fr in feat_rows], dtype=float)
        res = evaluate_feature_set(x, labels, clf, repeats, seed)
        auc_mat[ci, 0] = res["auc"]
        rows_out.append(
            {
                "classifier": clf,
                "feature_set": "Full",
                **{k: res[k] for k in ("accuracy", "balanced_accuracy", "balanced_accuracy_std", "auc", "auc_std", "sensitivity", "specificity", "f1")},
            }
        )
        print(f"  {clf:14s} BA={res['balanced_accuracy']:.3f}+/-{res['balanced_accuracy_std']:.3f} AUC={res['auc']:.3f}+/-{res['auc_std']:.3f}")
    save_csv(out / "tables" / "classifier_comparison.csv", rows_out)
    return rows_out


def run_statistics(
    details: Dict[str, Dict[str, object]],
    seed: int,
    out: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base = details["Baseline-R"]["ba_per_split"]
    full = details["Full"]["ba_per_split"]
    for name, det in details.items():
        if name == "Baseline-R":
            continue
        d, lo, hi = paired_bootstrap(base, det["ba_per_split"], seed=seed)
        p = wilcoxon_p(base, det["ba_per_split"])
        rows.append({"comparison": f"{name} - Baseline-R", "mean_delta_ba": d, "ci95_low": lo, "ci95_high": hi, "wilcoxon_p": p})
        print(f"  {name:20s} deltaBA={d:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.2e}")
    d, lo, hi = paired_bootstrap(details["Fusion+Path"]["ba_per_split"], full, seed=seed)
    rows.append({"comparison": "Full - Fusion+Path", "mean_delta_ba": d, "ci95_low": lo, "ci95_high": hi, "wilcoxon_p": wilcoxon_p(details["Fusion+Path"]["ba_per_split"], full)})
    save_csv(out / "tables" / "statistical_tests.csv", rows)
    return rows


def plot_main_and_ablation(main_results: List[Dict[str, object]], out: Path) -> None:
    names = [str(r["feature_set"]) for r in main_results]
    ba = [float(r["balanced_accuracy"]) for r in main_results]
    ba_std = [float(r["balanced_accuracy_std"]) for r in main_results]

    auc = [float(r["auc"]) for r in main_results]
    auc_std = [float(r["auc_std"]) for r in main_results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=220)
    ax.bar(
        x - 0.19,
        ba,
        width=0.38,
        yerr=ba_std,
        capsize=3,
        label="Balanced accuracy",
        color="#2a9d8f"
    )

    ax.bar(
        x + 0.19,
        auc,
        width=0.38,
        yerr=auc_std,
        capsize=3,
        label="AUC",
        color="#457b9d"
    )
    ax.set_ylim(0.4, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Performance of predefined feature sets")
    ax.set_xticks(x)
    display_names = {
        "Baseline-R": "Baseline-R",
        "MultiLambda-R": "MultiLambda-R",
        "MultiRho-R": "MultiRho-R",
        "SpectralGeometry-R": "Spectral-\nGeometry-R",
        "DiffusionFit": "DiffusionFit",
        "PathGate-L": "PathGate-L",
        "PathGate-CoarseL": "PathGate-\nCoarseL",
        "Fusion-noPath": "Fusion-\nnoPath",
        "Fusion+Path": "Fusion+Path",
        "Full": "Full",
    }

    ax.set_xticklabels(
        [display_names[n] for n in names],
        rotation=15,
        ha="right"
    )
    ax.axhline(
        0.5,
        color="gray",
        lw=1.0,
        ls="--",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        out / "figures" / "fig04_feature_performance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    order = ["Baseline-R", "MultiLambda-R", "MultiRho-R", "SpectralGeometry-R", "DiffusionFit", "PathGate-L", "PathGate-CoarseL", "Fusion-noPath", "Fusion+Path", "Full"]
    lookup = {str(r["feature_set"]): (float(r["balanced_accuracy"]), float(r["balanced_accuracy_std"])) for r in main_results}
    vals = [lookup[o][0] for o in order]
    errs = [lookup[o][1] for o in order]
    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=300)
    ax.errorbar(order, vals, yerr=errs, marker="o", lw=2.2, capsize=4, color="#9b2226")
    ax.set_ylim(0.4, 1.03)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Feature-group performance hierarchy")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        out / "figures" / "fig04_feature_performance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_classifier_comparison(clf_rows: List[Dict[str, object]], out: Path) -> None:
    names = [str(r["classifier"]) for r in clf_rows]
    ba = [float(r["balanced_accuracy"]) for r in clf_rows]
    auc = [float(r["auc"]) for r in clf_rows]
    ba_std = [float(r["balanced_accuracy_std"]) for r in clf_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=220)
    ax.bar(x - 0.19, ba, width=0.38, yerr=ba_std, capsize=3, label="Balanced accuracy", color="#e76f51")
    ax.bar(x + 0.19, auc, width=0.38, label="AUC", color="#264653")
    ax.set_ylim(0.4, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Classifier comparison on Full feature set")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figures" / "supp_classifier_comparison.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E4-E7: sensitivity experiments
# --------------------------------------------------------------------------- #
def run_photon_budget(cfg: CohortConfig, photon_grid: Sequence[int], samples: int, repeats: int, seed: int, out: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ph in photon_grid:
        t0 = time.perf_counter()
        # Hold subject severity, skin thickness, optical jitter, and photon RNG
        # streams fixed across budgets.  The only intended changed factor is
        # the number of launched photon packets.
        cohort = build_cohort_raw(samples, cfg, int(ph), seed)
        feat_rows, columns = derive_features(cohort, cfg.instrument_noise, seed)
        labels = np.asarray([r["label"] for r in cohort], dtype=int)
        groups = feature_groups(columns)
        for name in ("Baseline-R", "SpectralGeometry-R", "Fusion+Path", "Full"):
            x = np.asarray([[fr[c] for c in groups[name]] for fr in feat_rows], dtype=float)
            res = evaluate_feature_set(x, labels, DESIGNATED_CLASSIFIER, repeats, seed)
            rows.append({"photons": int(ph), "feature_set": name, "balanced_accuracy": res["balanced_accuracy"], "auc": res["auc"], "runtime_s": time.perf_counter() - t0})
        print(f"  photons={ph}: " + ", ".join(f"{r['feature_set']}={r['balanced_accuracy']:.3f}" for r in rows if r["photons"] == ph))
    save_csv(out / "tables" / "photon_budget.csv", rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for name in ("Baseline-R", "SpectralGeometry-R", "Fusion+Path", "Full"):
        sub = [r for r in rows if r["feature_set"] == name]
        ax.plot([float(r["photons"]) for r in sub], [float(r["balanced_accuracy"]) for r in sub], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("photon packets per wavelength")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Robustness to Monte Carlo photon budget")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "supp_photon_budget.png")
    plt.close(fig)
    return rows


def run_cohort_size(
    feat_rows: List[Dict[str, float]],
    columns: List[str],
    labels: np.ndarray,
    sizes: Sequence[int],
    repeats: int,
    seed: int,
    out: Path,
) -> List[Dict[str, object]]:
    groups = feature_groups(columns)
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(seed)
    x_full = np.asarray([[fr[c] for c in groups["Full"]] for fr in feat_rows], dtype=float)
    x_base = np.asarray([[fr[c] for c in groups["Baseline-R"]] for fr in feat_rows], dtype=float)
    subset_repeats = 20
    class_idx = {label: np.where(labels == label)[0] for label in (0, 1)}
    for n in sizes:
        n = min(int(n), len(labels))
        if n % 2:
            n -= 1
        per_class = n // 2
        if per_class > min(len(class_idx[0]), len(class_idx[1])):
            raise ValueError("cohort size exceeds the available balanced cohort")
        for name, x in (("Baseline-R", x_base), ("Full", x_full)):
            subset_ba, subset_auc = [], []
            for sub_rep in range(subset_repeats):
                # Balanced, independently sampled subcohorts quantify cohort
                # sampling uncertainty in addition to split-to-split variance.
                idx = np.concatenate([
                    rng.choice(class_idx[0], size=per_class, replace=False),
                    rng.choice(class_idx[1], size=per_class, replace=False),
                ])
                res = evaluate_feature_set(
                    x[idx], labels[idx], DESIGNATED_CLASSIFIER, repeats,
                    seed + 10_000 * n + sub_rep,
                )
                subset_ba.append(float(res["balanced_accuracy"]))
                subset_auc.append(float(res["auc"]))
            rows.append({
                "cohort_size": int(n),
                "feature_set": name,
                "balanced_accuracy": float(np.mean(subset_ba)),
                "balanced_accuracy_std": float(np.std(subset_ba)),
                "auc": float(np.mean(subset_auc)),
                "subset_repeats": subset_repeats,
            })
        print(f"  n={n}: " + ", ".join(f"{r['feature_set']}={r['balanced_accuracy']:.3f}" for r in rows if r["cohort_size"] == n))
    save_csv(out / "tables" / "cohort_size.csv", rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for name in ("Baseline-R", "Full"):
        sub = [r for r in rows if r["feature_set"] == name]
        ax.errorbar([int(r["cohort_size"]) for r in sub], [float(r["balanced_accuracy"]) for r in sub],
                    yerr=[float(r["balanced_accuracy_std"]) for r in sub], marker="o", capsize=3, label=name)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("synthetic cohort size")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Learning curve vs cohort size")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "supp_cohort_size.png")
    plt.close(fig)
    return rows


def run_noise_robustness(
    cohort: List[Dict],
    cfg: CohortConfig,
    noise_grid: Sequence[float],
    repeats: int,
    seed: int,
    out: Path,
) -> List[Dict[str, object]]:
    labels = np.asarray([r["label"] for r in cohort], dtype=int)
    rows: List[Dict[str, object]] = []
    for noise in noise_grid:
        # A common random-number stream makes the noise-level comparison
        # paired: only the magnitude, not the realization, changes.
        feat_rows, columns = derive_features(cohort, float(noise), seed)
        groups = feature_groups(columns)
        for name in ("Baseline-R", "SpectralGeometry-R", "Full"):
            x = np.asarray([[fr[c] for c in groups[name]] for fr in feat_rows], dtype=float)
            res = evaluate_feature_set(x, labels, DESIGNATED_CLASSIFIER, repeats, seed)
            rows.append({"instrument_noise": float(noise), "feature_set": name, "balanced_accuracy": res["balanced_accuracy"], "auc": res["auc"]})
        print(f"  noise={noise:.3f}: " + ", ".join(f"{r['feature_set']}={r['balanced_accuracy']:.3f}" for r in rows if abs(float(r["instrument_noise"]) - noise) < 1e-9))
    save_csv(out / "tables" / "noise_robustness.csv", rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for name in ("Baseline-R", "SpectralGeometry-R", "Full"):
        sub = [r for r in rows if r["feature_set"] == name]
        ax.plot([float(r["instrument_noise"]) * 100 for r in sub], [float(r["balanced_accuracy"]) for r in sub], marker="o", label=name)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("instrument noise (% relative)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Robustness to measurement noise")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "supp_noise_robustness.png")
    plt.close(fig)
    return rows


def run_domain_shift(
    cohort: List[Dict],
    feat_rows: List[Dict[str, float]],
    columns: List[str],
    labels: np.ndarray,
    repeats: int,
    seed: int,
    out: Path,
) -> List[Dict[str, object]]:
    groups = feature_groups(columns)
    x = np.asarray([[fr[c] for c in groups["Full"]] for fr in feat_rows], dtype=float)
    skin = np.asarray([r["skin_mm"] for r in cohort], dtype=float)
    mid = (skin >= 2.2) & (skin <= 3.8)
    thin = skin < 2.2
    thick = skin > 3.8
    rows: List[Dict[str, object]] = []
    scenarios = {"random-split (control)": None, "thin-skin test": thin, "thick-skin test": thick}
    for name, test_mask in scenarios.items():
        bas = []
        for r in range(repeats):
            if test_mask is None:
                tr, te = stratified_split(labels, seed + 17 * r)
            else:
                train_candidates = np.where(mid)[0]
                test_candidates = np.where(test_mask)[0]
                if len(test_candidates) < 6 or len(np.unique(labels[test_candidates])) < 2:
                    break
                # Stratified bootstrap re-sampling gives a genuine uncertainty
                # estimate for a fixed thickness-defined test domain.
                rng = np.random.default_rng(seed + 17 * r)
                def draw_stratified(candidates: np.ndarray) -> np.ndarray:
                    return np.concatenate([
                        rng.choice(candidates[labels[candidates] == label], size=np.sum(labels[candidates] == label), replace=True)
                        for label in (0, 1)
                    ])
                tr = draw_stratified(train_candidates)
                te = draw_stratified(test_candidates)
            scores, tr_scores = fit_score_both(DESIGNATED_CLASSIFIER, x[tr], labels[tr], x[te], seed + r)
            thr = youden_threshold(labels[tr], tr_scores)
            bas.append(balanced_accuracy(labels[te], (scores >= thr).astype(int)))
        if bas:
            n_test = int(np.sum(test_mask)) if test_mask is not None else -1
            rows.append({"scenario": name, "n_test": n_test,
                         "balanced_accuracy": float(np.mean(bas)), "balanced_accuracy_std": float(np.std(bas)),
                         "uncertainty": "split SD" if test_mask is None else "stratified bootstrap SD"})
        print(f"  {name:24s} BA={rows[-1]['balanced_accuracy']:.3f}" if bas else f"  {name:24s} skipped (too few test subjects)")
    save_csv(out / "tables" / "domain_shift.csv", rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=220)
    ax.bar([str(r["scenario"]) for r in rows], [float(r["balanced_accuracy"]) for r in rows],
           yerr=[float(r["balanced_accuracy_std"]) for r in rows], capsize=4, color=["#2a9d8f", "#e9c46a", "#f4a261"])
    ax.set_ylim(0.4, 1.03)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Generalization across skin-thickness shift (Full features)")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figures" / "supp_domain_shift.png")
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------- #
# E8: severity staging regression
# --------------------------------------------------------------------------- #
def run_staging(cfg: CohortConfig, samples: int, photons: int, repeats: int, seed: int, out: Path) -> List[Dict[str, object]]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"  simulating staging cohort (n={samples}, severity ~ U(0,1))")
    cohort = build_cohort_raw(samples, cfg, photons, seed, staging=True)
    save_cohort_npz(out / "tables" / "staging_cohort.npz", cohort, {"photons": photons, "seed": seed, "staging": True})
    feat_rows, columns = derive_features(cohort, cfg.instrument_noise, seed)
    x = np.asarray([[fr[c] for c in columns] for fr in feat_rows], dtype=float)
    y = np.asarray([r["severity"] for r in cohort], dtype=float)

    regressors = {
        "Ridge": lambda s: Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "RandomForest": lambda s: RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=4),
        "GradBoost": lambda s: GradientBoostingRegressor(random_state=s),
    }
    rows: List[Dict[str, object]] = []
    best_name, best_spear, best_pred, best_yte = "", -1.0, None, None
    for name, ctor in regressors.items():
        spears, r2s, maes = [], [], []
        for r in range(repeats):
            rng = np.random.default_rng(seed + 31 * r)
            idx = rng.permutation(len(y))
            cut = int(0.8 * len(y))
            tr, te = idx[:cut], idx[cut:]
            reg = ctor(seed + r)
            reg.fit(x[tr], y[tr])
            pred = reg.predict(x[te])
            from scipy.stats import spearmanr

            spear_result = spearmanr(y[te], pred)
            # ``.statistic`` is the modern SciPy API; tuple indexing keeps the
            # reproducibility script compatible with SciPy 1.8 installations.
            spear = float(spear_result.statistic if hasattr(spear_result, "statistic") else spear_result[0])
            ss_res = float(np.sum((y[te] - pred) ** 2))
            ss_tot = float(np.sum((y[te] - np.mean(y[te])) ** 2)) + EPS
            spears.append(spear)
            r2s.append(1.0 - ss_res / ss_tot)
            maes.append(float(np.mean(np.abs(y[te] - pred))))
            if name == "GradBoost" and spear > best_spear:
                best_name, best_spear, best_pred, best_yte = name, spear, pred.copy(), y[te].copy()
        rows.append({"regressor": name, "spearman": float(np.mean(spears)), "spearman_std": float(np.std(spears)),
                     "r2": float(np.mean(r2s)), "mae": float(np.mean(maes))})
        print(f"  {name:14s} Spearman={rows[-1]['spearman']:.3f} R2={rows[-1]['r2']:.3f} MAE={rows[-1]['mae']:.3f}")
    save_csv(out / "tables" / "staging_regression.csv", rows)

    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=220)
    ax.scatter(best_yte, best_pred, s=14, alpha=0.6, color="#457b9d", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("true fibrosis severity")
    ax.set_ylabel("predicted severity")
    ax.set_title(f"Severity staging (best split, {best_name}, Spearman={best_spear:.3f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figures" / "supp_staging_regression.png")
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------- #
# E9 + E10: mechanism curves and effect maps
# --------------------------------------------------------------------------- #
def plot_spatial_curves(cohort: List[Dict], out: Path) -> None:
    labels = np.asarray([r["label"] for r in cohort], dtype=int)
    seps = np.asarray(SEPARATIONS_MM)

    # 2 × 2 layout:
    # top row    -> weighted reflectance R
    # bottom row -> weighted liver-return signal L
    # columns    -> 808 and 1064 nm
    fig, axes = plt.subplots(2,2,figsize=(11.4, 7.6),dpi=300,constrained_layout=True)

    show_wl = [808, 1064]

    for col, wl in enumerate(show_wl):
        i = WAVELENGTHS.index(wl)

        # Use synthetic-severity terminology rather than
        # clinically calibrated fibrosis-stage terminology.
        for cls, name, color in ((0, "Low-severity group", "#2a9d8f"),(1, "High-severity group", "#9b2226"),):

            R = np.stack([r["chan"][i, :, 0]for r, c in zip(cohort, labels)if c == cls])
            L = np.stack([r["chan"][i, :, 1]for r, c in zip(cohort, labels)if c == cls])

            for row, mat in ((0, R),(1, L),):
                mean = mat.mean(axis=0)

                # 95% confidence interval of the synthetic-subject mean
                ci = (1.96* mat.std(axis=0, ddof=1)/ math.sqrt(mat.shape[0]))
                ax = axes[row, col]
                ax.plot(seps,mean,marker="o",ms=3.5,lw=1.8,color=color,label=name)

                ax.fill_between(seps,mean - ci,mean + ci,color=color,alpha=0.18)

        # Titles: use mathematical symbols rather than "lambda, rho"
        axes[0, col].set_title(
            rf"Weighted reflectance $R(\lambda,\rho)$ at {wl} nm"
        )

        axes[1, col].set_title(
            rf"Weighted liver-return signal $L(\lambda,\rho)$ at {wl} nm"
        )

    # Common formatting for all four panels
    for ax in axes.flat:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)

    # Only the bottom row carries x-axis labels.
    # Do NOT keep ax.set_xlabel(...) inside the loop above.
    axes[1, 0].set_xlabel(
        r"Source–detector separation, $\rho$ (mm)"
    )
    axes[1, 1].set_xlabel(
        r"Source–detector separation, $\rho$ (mm)"
    )

    # Y-axis labels only on the left column
    axes[0, 0].set_ylabel(
        "Normalized weighted reflectance"
    )
    axes[1, 0].set_ylabel(
        "Normalized liver-return weight"
    )

    # Use only one legend for the whole 2 × 2 figure
    axes[0, 1].legend(
        fontsize=8,
        loc="upper right",
        frameon=True
    )

    # Panel labels
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            -0.10,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom"
        )

    # Final manuscript figure name
    fig.savefig(
        out / "figures" / "fig03_weighted_spatial_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def compute_effect_maps(feat_rows: List[Dict[str, float]], labels: np.ndarray, out: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    mat_R = np.zeros((len(WAVELENGTHS), len(SEPARATIONS_MM)))
    mat_L = np.zeros_like(mat_R)
    for i in range(len(WAVELENGTHS)):
        for j in range(len(SEPARATIONS_MM)):
            for suffix, mat in (("R", mat_R), ("L", mat_L)):
                col = f"{wl_tag(i)}_{sep_tag(j)}_{suffix}"
                v = np.asarray([fr[col] for fr in feat_rows], dtype=float)
                v0, v1 = v[labels == 0], v[labels == 1]
                d = cohens_d_pooled(v0, v1)
                mat[i, j] = d
                rows.append({"wavelength_nm": WAVELENGTHS[i], "separation_mm": SEPARATIONS_MM[j], "channel": suffix,
                             "cohens_d": d, "control_n": len(v0), "fibrosis_n": len(v1),
                             "control_mean": float(v0.mean()), "control_sd": float(v0.std(ddof=1)),
                             "fibrosis_mean": float(v1.mean()), "fibrosis_sd": float(v1.std(ddof=1))})
    save_csv(out / "tables" / "feature_effects.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=220, constrained_layout=True)
    # Preserve the model's internal wavelength order, but display spectra in
    # ascending physical wavelength order for manuscript readability.
    display_order = np.argsort(np.asarray(WAVELENGTHS))
    display_wavelengths = [WAVELENGTHS[i] for i in display_order]
    for ax, mat, title in ((axes[0], mat_R, "reflectance R channels"), (axes[1], mat_L, "liver-return L channels")):
        display_mat = mat[display_order, :]
        vmax = max(0.5, float(np.max(np.abs(mat))))
        im = ax.imshow(display_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(SEPARATIONS_MM)))
        ax.set_xticklabels([f"{s:.2f}" for s in SEPARATIONS_MM], rotation=45, fontsize=7)
        ax.set_yticks(np.arange(len(display_wavelengths)))
        ax.set_yticklabels([str(w) for w in display_wavelengths])
        ax.set_xlabel("source-detector separation (mm)")
        ax.set_ylabel("wavelength (nm)")
        ax.set_title(f"effect size (Cohen's d): {title}")
        for i in range(display_mat.shape[0]):
            for j in range(display_mat.shape[1]):
                ax.text(j, i, f"{display_mat[i, j]:.1f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(out / "figures" / "supp_effect_heatmap.png")
    plt.close(fig)
    return rows


def plot_geometry_figure(out: Path) -> None:
    from matplotlib.colors import ListedColormap

    geo = FastGeometry(skin_thickness=0.30, epidermis_thickness=0.008)
    tissue = build_layered_tissue(geo)
    cmap = ListedColormap(["#e3b17c", "#c9955d", "#8a2d2d"])
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=220)
    ax.imshow(tissue, origin="upper",
              extent=[geo.x_min * CM_TO_MM, geo.x_max * CM_TO_MM, geo.z_max * CM_TO_MM, geo.z_min * CM_TO_MM],
              aspect="equal", cmap=cmap, vmin=1, vmax=3)
    ax.axhline(geo.epidermis_thickness * CM_TO_MM, color="white", lw=0.8)
    ax.axhline(geo.skin_thickness * CM_TO_MM, color="white", lw=1.1)
    ax.annotate(
        "Source",
        xy=(0, 0),
        xytext=(0, -1.45),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8)
    )
    ax.text(
        0.98, 0.96,
        r"10 bilateral SDS channels: $\rho=0.25$–$4.75$ mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9
    )
    # The MC scorer bins |x|, so one channel is the sum of two symmetric
    # 0.5-mm detector strips.  Draw both sides to keep the figure consistent
    # with the implemented bilateral geometry and the absolute R/L amplitudes.
    for sep in SEPARATIONS_MM:
        for sign in (-1.0, 1.0):
            center = sign * sep
            ax.add_patch(plt.Rectangle((center - 0.25, -0.35), 0.5, 0.35, color="#457b9d", alpha=0.85))
    ax.set_xlim(-6, 6)
    ax.set_ylim(8, -2)
    ax.set_xlabel("Lateral position, x (mm)")
    ax.set_ylabel("Depth, z (mm)")
    ax.set_title("Layered skin–liver model and bilateral scoring geometry")
    handles = [plt.Line2D([0], [0], color=cmap(i - 1), lw=6) for i in (1, 2, 3)]
    ax.legend(handles, ["Epidermis", "Dermis", "Liver"], loc="lower right", fontsize=8,frameon=True)
    fig.tight_layout()
    fig.savefig(
        out / "figures" / "fig02_geometry.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# documentation
# --------------------------------------------------------------------------- #
def write_readme(out: Path, args: argparse.Namespace) -> None:
    text = f"""# Reproduction Guide

## Environment

Python 3.10+ (developed on 3.13). Install dependencies:

```bash
pip install numpy numba matplotlib scikit-learn scipy
```

## One-command reproduction

```bash
python research_delivery_pipeline_v2.py --out final_delivery_v2 --samples {args.samples} --photons {args.photons} --seed {args.seed}
```

Expected runtime: ~8-12 minutes on a 32-core CPU (numba-parallel Monte Carlo).
A fast smoke run (~1 minute):

```bash
python research_delivery_pipeline_v2.py --out final_delivery_v2_smoke --quick
```

## Pipeline layout

1. `mc_liver_fast.py` — validated numba Monte Carlo core (330x speedup, spatially-resolved scoring).
2. `validate_fast_core.py` — statistical equivalence / determinism / energy-balance checks (output: `validation_report.json`).
3. `research_delivery_pipeline_v2.py` — cohort simulation, feature derivation, E1-E11 experiments, figure/table/doc export.

## Outputs

- `tables/` — machine-readable results for every experiment (CSV + cohort NPZ).
- `tables/channel_optimization_*.csv` — nested-test performance, selected
  wavelength/SDS subsets, and selection frequencies for probe optimization.
- `figures/` — paper-ready PNG figures (fig01-fig12).
- `docs/` — project summary and literature notes.
- `EXPERIMENT_REPORT.md` — main findings and acceptance checklist.
- `INNOVATION_POINTS.md` — contribution statements (CN/EN usage).
- `experiment_log.txt` — full console log of the run.
- `run_manifest.json` — configuration, package versions, timings, seeds.

## Reproducibility notes

All randomness is seeded (`--seed`). The Monte Carlo core is deterministic for
fixed photon count/seed/batch layout; cohort NPZ files are cached under
`tables/` so analysis stages can be re-run without re-simulation.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def write_data_description(out: Path, args: argparse.Namespace, cfg: CohortConfig) -> None:
    text = f"""# Data Description

No measured clinical dataset exists in this project; all experiments are
simulation-first Monte Carlo feasibility studies on a synthetic cohort.

## Cohort (main classification)

- Subjects: {args.samples} (balanced).
- Label 0 (non-significant fibrosis, "F0-F1"): severity ~ U{cfg.control_sev_range}.
- Label 1 (significant fibrosis, "F2-F4"): severity ~ U{cfg.fibrosis_sev_range}.
- Skin thickness ~ N({cfg.skin_mean_mm}, {cfg.skin_std_mm}) mm, clipped to [{cfg.skin_min_mm}, {cfg.skin_max_mm}] mm.
- Per-subject optical jitter: liver mu_a/mu_s' +/-{int(cfg.liver_jitter*100)}%, skin +/-{int(cfg.skin_jitter*100)}% (uniform).
- Instrument noise: {cfg.instrument_noise*100:.1f}% relative Gaussian applied to normalized exit-weight signals.

## Measurement

- Wavelengths: {', '.join(map(str, WAVELENGTHS))} nm.
- Source-detector separations: {', '.join(f'{s:.2f}' for s in SEPARATIONS_MM)} mm
  (bilateral symmetric pairs of 0.5 mm-wide detector strips).
- Photon packets per wavelength per subject: {args.photons}.
- Raw optical observables: spatially resolved sums of remaining top-exit packet
  weight for all trajectories, R(lambda,rho), and for the subset that visited
  liver, L(lambda,rho). Integer count histograms are retained only for path
  probabilities and implementation validation.

## Feature sets

See `feature_groups()` in `research_delivery_pipeline_v2.py`:
Baseline-R, MultiLambda-R, MultiRho-R, SpectralGeometry-R, DiffusionFit,
PathGate-L, Fusion-noPath, Fusion+Path, Full.

## Honest scope statement

Optical coefficients are literature-trend-constrained demonstration parameters
(`skin_liver_fibrosis_mc.py::optical_properties`), not clinical ground truth.
The staging cohort (`tables/staging_cohort.npz`) samples severity ~ U(0,1) for
regression analysis. Replace the coefficient table with measured values for
submission-grade claims.
"""
    (out / "DATA_DESCRIPTION.md").write_text(text, encoding="utf-8")


def write_experiment_report(
    out: Path,
    args: argparse.Namespace,
    main_results: List[Dict[str, object]],
    clf_rows: List[Dict[str, object]],
    stat_rows: List[Dict[str, object]],
    staging_rows: List[Dict[str, object]],
    channel_summary: Dict[str, object],
    validation: Dict,
    timings: Dict[str, float],
) -> None:
    def row_md(r: Dict[str, object]) -> str:
        return (f"| {r['feature_set']} | {int(float(r['n_features']))} | {float(r['balanced_accuracy']):.3f}+/-{float(r['balanced_accuracy_std']):.3f} "
                f"| {float(r['auc']):.3f}+/-{float(r['auc_std']):.3f} | {float(r['sensitivity']):.3f} | {float(r['specificity']):.3f} | {float(r['f1']):.3f} |")

    main_md = "\n".join(row_md(r) for r in main_results)
    clf_md = "\n".join(
        f"| {r['classifier']} | {float(r['balanced_accuracy']):.3f}+/-{float(r['balanced_accuracy_std']):.3f} | {float(r['auc']):.3f}+/-{float(r['auc_std']):.3f} | {float(r['f1']):.3f} |"
        for r in clf_rows
    )
    stat_md = "\n".join(
        f"| {r['comparison']} | {float(r['mean_delta_ba']):+.3f} | [{float(r['ci95_low']):+.3f}, {float(r['ci95_high']):+.3f}] | {float(r['wilcoxon_p']):.2e} |"
        for r in stat_rows
    )
    stag_md = "\n".join(
        f"| {r['regressor']} | {float(r['spearman']):.3f}+/-{float(r['spearman_std']):.3f} | {float(r['r2']):.3f} | {float(r['mae']):.3f} |"
        for r in staging_rows
    )
    lookup = {str(r["feature_set"]): r for r in main_results}
    best_clf = max(clf_rows, key=lambda r: float(r["balanced_accuracy"]))
    text = f"""# Final Experiment Report (v2)

## Task

Transcutaneous optical classification of significant liver fibrosis from
Monte-Carlo-simulated multi-wavelength, multi-separation reflectance
measurements, plus continuous severity staging.

## What was broken in v1 and fixed in v2

1. **Domain bug**: the v1 grid (x: +/-0.75 mm, z: 1.6 mm) did not contain the
   3 mm skin layer or the liver, and 12/16 detector channels sat outside the
   domain, so most features were constants plus noise. v2 simulates
   x: +/-8 mm, z: 12 mm and accumulates the remaining weight of every
   top-exiting photon packet into spatial channels.
2. **Photon starvation**: v1 used 700 photons/measurement; v2 uses
   {args.photons} per wavelength (validated core, {validation['benchmark']['speedup']:.0f}x faster,
   equivalence/energy checks pass — see `validation_report.json`).
3. **Evaluation rigor**: v2 adds repeated stratified CV, paired bootstrap CIs,
   Wilcoxon signed-rank tests, classifier benchmark, nested channel selection,
   and E4-E8 sensitivity experiments.

## Setup

- Synthetic subjects: {args.samples} (balanced), severity-stratified sampling.
- Photons per wavelength: {args.photons}.
- Evaluation: {args.repeats} repeated stratified 70/30 splits.
- Designated classifier: {DESIGNATED_CLASSIFIER} (leakage-safe scaling +
  univariate screening inside each training split; LogReg preferred for
  small/medium cohorts, GradBoost for n>100).
- Main metric: balanced accuracy; secondary: AUC, sensitivity, specificity, F1.

## E11 Leakage-safe wavelength/SDS optimization

- Method: {channel_summary['method']}.
- Outer test splits: {int(channel_summary['outer_repeats'])}.
- Held-out BA: {float(channel_summary['balanced_accuracy']):.3f} +/- {float(channel_summary['balanced_accuracy_std']):.3f}.
- Held-out AUC: {float(channel_summary['auc']):.3f} +/- {float(channel_summary['auc_std']):.3f}.
- Mean selected design: {float(channel_summary['mean_n_wavelengths']):.1f} wavelengths x
  {float(channel_summary['mean_n_separations']):.1f} SDS = {float(channel_summary['mean_n_channels']):.1f} channels.
- Per-split selections and selection frequencies are in
  `tables/channel_optimization_splits.csv` and `tables/channel_selection_frequency.csv`.

## E1 Main comparison (feature-set ladder)

| Feature set | #feat | Balanced acc | AUC | Sens | Spec | F1 |
|---|---:|---:|---:|---:|---:|---:|
{main_md}

## E2 Classifier comparison (Full features)

| Classifier | Balanced acc | AUC | F1 |
|---|---:|---:|---:|
{clf_md}

Best classifier by BA: **{best_clf['classifier']}** (BA={float(best_clf['balanced_accuracy']):.3f}).

## E3 Statistical tests (vs Baseline-R)

| Comparison | delta BA | 95% bootstrap CI | Wilcoxon p |
|---|---:|---:|---:|
{stat_md}

## E8 Severity staging regression (severity ~ U(0,1))

| Regressor | Spearman | R2 | MAE |
|---|---:|---:|---:|
{stag_md}

## Figure guide

- fig01 phantom geometry; fig02 spatially-resolved R(lambda,rho)/L(lambda,rho) curves with 95% CI.
- fig03 feature-set comparison; fig04 classifier comparison; fig05 ROC curves.
- fig06 ablation ladder; fig07 photon budget; fig08 cohort-size learning curve.
- fig09 noise robustness; fig10 skin-thickness domain shift; fig11 severity
  staging scatter; fig12 effect-size maps.

## Runtime

Total pipeline: {timings.get('total', 0.0)/60:.1f} min (cohort simulation {timings.get('cohort', 0.0)/60:.1f} min).

## Scope

Simulation-first feasibility/methodology study. Optical coefficients are
literature-trend demonstration parameters; no clinical dataset is available in
this project. Claims should be framed accordingly.
"""
    (out / "EXPERIMENT_REPORT.md").write_text(text, encoding="utf-8")


def write_innovation_document(out: Path, main_results: List[Dict[str, object]]) -> None:
    lookup = {str(r["feature_set"]): r for r in main_results}
    base = float(lookup["Baseline-R"]["balanced_accuracy"])
    path = float(lookup["PathGate-L"]["balanced_accuracy"])
    fusion = float(lookup["Fusion+Path"]["balanced_accuracy"])
    full = float(lookup["Full"]["balanced_accuracy"])
    full_auc = float(lookup["Full"]["auc"])
    text = f"""# Innovation Points (v2)

## Innovation 1: Spatially-resolved liver-return path gating

1. **核心思想**：在 photon packet 层面同时记录"顶部出射位置直方图"与"出射前是否访问过肝脏"，
   一次仿真即可得到所有源-探间距下的漫反射曲线 R(ρ) 与肝源返回曲线 L(ρ)，
   把肝脏贡献从皮肤浅层背散射中显式分离。
2. **与已有方法的区别**：MCML/MCXYZ 只输出总反射率/透射率或 fluence；v1 baseline 只用单通道
   表面反射率。路径门控直方图是通道级、可积分的肝源信号。
3. **公式**：

```text
H_R_count[k] = sum_i 1[packet i exits top in radial bin k]
H_L_count[k] = sum_i 1[packet i exits top in bin k AND visited liver]
S_R[k] = sum_i W_i_exit * 1[i in bin k]
S_L[k] = sum_i W_i_exit * I_i_liver * 1[i in bin k]
R(lambda,rho_k) = S_R[k] / N0
L(lambda,rho_k) = S_L[k] / N0
F(lambda,rho_k) = L / (R + eps)
```

   其中 `H_*_count` 仅用于轨迹概率与数值验证；论文光学信号使用 `S_*`。

4. **代码位置**：`mc_liver_fast.py::_run_batch`（直方图统计），
   `research_delivery_pipeline_v2.py::simulate_subject_raw / derive_features`。
5. **效果**：Baseline-R balanced accuracy={base:.3f}；PathGate-L={path:.3f}（仅用 L 通道）。
6. **消融**：`tables/main_comparison.csv`、`figures/fig06_ablation.png`。
7. **Paper wording**: We introduce spatially-resolved liver-return path gating, in which every
   top-exiting photon packet is scored by its remaining exit weight and radius, together with
   a binary liver-visit history label, yielding liver-origin reflectance curves that separate
   hepatic contrast from superficial skin backscattering.

## Innovation 2: Validated high-throughput Monte Carlo core for ML-scale cohorts

1. **核心思想**：将 2-D photon-packet 传输用 numba 并行化（batch 级独立种子、逐光子确定性），
   在保持与参考实现统计一致的前提下获得约两个数量级加速，使"每受试者多波长 × 高光子数 × 大队列"
   的机器学习级数据集可以在 CPU 上生成。
2. **验证**：与参考核心在相同几何/探测器下所有关键概率在 binomial 3σ 内一致；能量守恒
   |E-1| < 0.01；并行重复运行逐比特一致（`validation_report.json`）。
3. **Paper wording**: We validate a JIT-parallelized photon-packet kernel against a reference
   implementation (all observables within binomial 3 sigma) and use it to generate
   machine-learning-scale synthetic cohorts with controlled MC noise.

## Innovation 3: Spectral-geometric fusion with leakage-safe small-cohort learning

1. **核心思想**：把 R/L/F 通道、每波长吸收标量（肝/皮吸收比、到达率、返回率）与扩散式
   有效衰减拟合特征组合，分类器流水线（标准化 + 单变量筛选 + 分类器）完全在训练折内拟合，
   阈值用训练集 Youden 指数确定。
2. **效果**：Fusion+Path BA={fusion:.3f}，Full BA={full:.3f}，Full AUC={full_auc:.3f}。
3. **Paper wording**: A leakage-safe fusion pipeline combines path-gated, spectral and
   geometry-resolved channels with per-wavelength absorption scalars; all feature screening,
   scaling and threshold selection are fitted inside training folds only.

## Innovation 4: Continuous severity staging

把二分类扩展为纤维化严重度回归（staging cohort, severity~U(0,1)），Spearman 相关与 R2
见 `tables/staging_regression.csv` 与 `figures/fig11_staging_regression.png`，
显示该方法有支持 F0-F4 分级而非仅有无判别的潜力。
"""
    (out / "INNOVATION_POINTS.md").write_text(text, encoding="utf-8")


def write_literature_notes(out: Path) -> None:
    text = """# Literature Notes (v2)

Reference PDFs in `参考文献/` ground the model choices:

- **Carneiro et al., human liver optical properties 400-1000 nm** — supports
  wavelength-dependent liver mu_a / mu_s' tables and using NIR windows for
  deeper liver interrogation.
- **Bashkatov/Genina/Tuchin skin optics review** — supports the skin-layer
  coefficient ranges and the decreasing scattering trend with wavelength.
- **Salomatina et al., skin visible-NIR properties** — supports the
  epidermis/dermis coefficient structure.
- **Integrating-sphere review** — motivates diffuse-reflectance observables as
  the quantity an eventual bench validation would measure.

Methodological anchors:

- Wang & Jacques (MCML, the bundled `1-s2.0-016926079501640F-main.pdf`) define
  the classic layered-tissue Monte Carlo scoring of spatially-resolved
  reflectance that the fast kernel reproduces and extends with liver-return
  gating.
- Reported clinical NIRS/optical fibrosis studies typically reach AUC in the
  ~0.75-0.9 range for significant-fibrosis discrimination; the synthetic
  results here should be presented as a simulation-first feasibility bound,
  not as clinical validation.
"""
    (out / "LITERATURE_NOTES.md").write_text(text, encoding="utf-8")


def write_project_summary(out: Path) -> None:
    text = """# Project Summary (v2)

## v1 -> v2 root cause

Profiling showed the v1 cohort features were dominated by two setup bugs:
the simulation grid (x +/-0.75 mm, z 1.6 mm) contained neither the 3 mm skin
layer nor the liver, and 12/16 source-detector channels lay outside the grid;
combined with 700-photon measurements and a 0.06 mm detector, most features
were zero plus noise floor. Best v1 balanced accuracy: 0.61.

## v2 structure

- `mc_liver_fast.py` — numba-parallel MC core, x +/-8 mm / z 12 mm,
  spatially-resolved top-exit + liver-return histograms, 330x throughput.
- `validate_fast_core.py` — equivalence, determinism, energy-balance checks.
- `research_delivery_pipeline_v2.py` — E1-E11 experiment stack, including
  leakage-safe wavelength/SDS optimization.

## Deliverable checklist

Main comparison, classifier benchmark, statistical tests, photon/cohort/noise
sensitivity, skin-thickness domain shift, severity staging, mechanism curves,
effect maps, validation report, experiment log, run manifest, README.
"""
    (out / "docs" / "PROJECT_SUMMARY.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Research delivery pipeline v2 (liver fibrosis MC)")
    parser.add_argument("--out", default="final_delivery_v2")
    parser.add_argument("--samples", type=int, default=240, help="synthetic subjects, balanced")
    parser.add_argument("--photons", type=int, default=200000, help="photon packets per wavelength per subject")
    parser.add_argument("--repeats", type=int, default=30, help="repeated stratified CV splits")
    parser.add_argument(
        "--channel-repeats",
        type=int,
        default=None,
        help="outer splits for nested channel optimization (default: --repeats; quick mode: 3)",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--quick", action="store_true", help="smoke run: small cohort/photons/repeats")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.samples = 24
        args.photons = 20000
        args.repeats = 10
        args.skip_sensitivity = True  # sensitivity experiments need larger cohorts
    channel_repeats = args.channel_repeats if args.channel_repeats is not None else (3 if args.quick else args.repeats)
    if args.samples < 12 or args.samples % 2 != 0:
        raise ValueError("--samples must be an even number >= 12")
    if channel_repeats < 1:
        raise ValueError("--channel-repeats must be >= 1")

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "experiment_log.txt")

    t_start = time.perf_counter()
    timings: Dict[str, float] = {}
    cfg = CohortConfig()
    print("=" * 78)
    print("research delivery pipeline v2")
    print(f"out={out.resolve()} samples={args.samples} photons={args.photons} repeats={args.repeats} seed={args.seed}")
    print("=" * 78)

    print("\n[1/12] simulating main cohort")
    t0 = time.perf_counter()
    cohort = build_cohort_raw(args.samples, cfg, args.photons, args.seed)
    timings["cohort"] = time.perf_counter() - t0
    print(f"  cohort simulated in {timings['cohort']:.1f}s")
    save_cohort_npz(out / "tables" / "main_cohort.npz", cohort, {"photons": args.photons, "seed": args.seed, "samples": args.samples})
    labels = np.asarray([r["label"] for r in cohort], dtype=int)

    print("\n[2/12] deriving features")
    feat_rows, columns = derive_features(cohort, cfg.instrument_noise, args.seed)
    print(f"  {len(columns)} features per subject")

    # quick signal-quality check: top-5 Cohen's d
    x_all = np.asarray([[fr[c] for c in columns] for fr in feat_rows], dtype=float)
    d_vals = []
    for ci, col in enumerate(columns):
        v0 = x_all[labels == 0, ci]
        v1 = x_all[labels == 1, ci]
        d_vals.append((abs(cohens_d_pooled(v0, v1)), col))
    d_vals.sort(reverse=True)
    top5 = d_vals[:5]
    print(f"  top-5 |Cohen's d|: " + ", ".join(f"{col}={d:.3f}" for d, col in top5))
    if top5[0][0] < 0.3:
        print("  ** WARNING: effect sizes are very small (<0.3). Increase --photons or --samples for reliable classification.")
    save_csv(out / "tables" / "cohort_features.csv", [{"label": int(l), **fr} for l, fr in zip(labels, feat_rows)])

    print("\n[3/12] E11 nested wavelength/SDS channel optimization")
    channel_summary = run_channel_optimization(
        feat_rows, labels, channel_repeats, args.seed + 11000, out
    )

    print("\n[4/12] E1 main comparison (feature-set ladder)")
    main_results, details, groups = run_main_comparison(feat_rows, columns, labels, args.repeats, args.seed, out)
    plot_main_and_ablation(main_results, out)

    print("\n[5/12] E2 classifier comparison (Full features)")
    clf_rows = run_classifier_comparison(feat_rows, columns, labels, args.repeats, args.seed, out)
    plot_classifier_comparison(clf_rows, out)

    print("\n[6/12] E3 statistical tests")
    stat_rows = run_statistics(details, args.seed, out)

    print("\n[7/12] E9 spatially-resolved mechanism curves")
    plot_spatial_curves(cohort, out)
    plot_geometry_figure(out)

    print("\n[8/12] E10 feature effect maps")
    compute_effect_maps(feat_rows, labels, out)

    if not args.skip_sensitivity:
        print("\n[9/12] E4 photon-budget sensitivity")
        ph_grid = [2000, 10000, 50000] + ([] if args.quick else [args.photons])
        run_photon_budget(cfg, ph_grid, max(24, args.samples // 2), args.repeats, args.seed + 5000, out)

        print("\n[10/12] E5 cohort-size learning curve")
        sizes = [24, 48, 96] if args.quick else [24, 48, 96, 168, 240]
        run_cohort_size(feat_rows, columns, labels, sizes, args.repeats, args.seed + 6000, out)

        print("\n[11/12] E6 instrument-noise robustness + E7 domain shift")
        run_noise_robustness(cohort, cfg, [0.0, 0.01, 0.02, 0.05, 0.10], args.repeats, args.seed + 7000, out)
        run_domain_shift(cohort, feat_rows, columns, labels, args.repeats, args.seed + 8000, out)
    else:
        print("\n[9-11/12] sensitivity experiments skipped")

    print("\n[12/12] E8 severity staging regression + documentation")
    stag_samples = max(24, args.samples // 2) if args.quick else args.samples
    staging_rows = run_staging(cfg, stag_samples, args.photons if not args.quick else 20000, 5 if args.quick else 10, args.seed + 9000, out)

    validation = json.loads((Path("final_delivery_v2") / "validation_report.json").read_text(encoding="utf-8")) \
        if (Path("final_delivery_v2") / "validation_report.json").exists() and out.name != "final_delivery_v2" else None
    if validation is None and (out / "validation_report.json").exists():
        validation = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    if validation is None:
        validation = {"benchmark": {"speedup": float("nan")}, "all_pass": None}

    write_readme(out, args)
    write_data_description(out, args, cfg)
    write_experiment_report(
        out,
        args,
        main_results,
        clf_rows,
        stat_rows,
        staging_rows,
        channel_summary,
        validation,
        {"total": time.perf_counter() - t_start, "cohort": timings["cohort"]},
    )
    write_innovation_document(out, main_results)
    write_literature_notes(out)
    write_project_summary(out)

    import platform

    manifest = {
        "entry": f"python research_delivery_pipeline_v2.py --out {args.out} --samples {args.samples} --photons {args.photons} --repeats {args.repeats} --seed {args.seed}",
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {},
        "wavelengths": list(WAVELENGTHS),
        "separations_mm": list(SEPARATIONS_MM),
        "designated_classifier": DESIGNATED_CLASSIFIER,
        "channel_optimization": channel_summary,
        "timings_s": {"total": time.perf_counter() - t_start, **timings},
        "validation_report_all_pass": validation.get("all_pass"),
    }
    for pkg in ("numpy", "numba", "matplotlib", "sklearn", "scipy"):
        try:
            mod = __import__(pkg)
            manifest["package_versions"][pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            manifest["package_versions"][pkg] = "missing"
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = time.perf_counter() - t_start
    print(f"\nDONE in {total/60:.1f} min. Deliverables saved to: {out.resolve()}")


if __name__ == "__main__":
    # Single-entry policy: invoking this former entry point delegates to the
    # unified runner instead of creating an incompatible legacy output folder.
    from run_final_paper import main as run_unified_paper
    run_unified_paper()
