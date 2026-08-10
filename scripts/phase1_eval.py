"""Phase 1 — shared cross-validation and comparison machinery.

The single most important property here: every model is scored on the SAME fold
assignments, generated once. That makes every comparison paired by construction,
so a delta can carry a confidence interval instead of two independent means being
eyeballed against each other.

Rules honoured (CLAUDE.md, docs/phase1-spec.md):
  - StratifiedGroupKFold, groups=sub_id. No subject in train and test together.
  - All preprocessing inside the fold, enforced by using a Pipeline. Nothing is
    fit outside cross_validate.
  - Repeated CV (5 folds x 5 seeds) because 5 estimates is too thin for a CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

N_SPLITS = 5
N_REPEATS = 5
SEEDS = [11, 23, 37, 53, 71]
INNER_SPLITS = 3
LOGIT_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def make_folds(y: np.ndarray, groups: np.ndarray) -> list[tuple]:
    """Generate the fold assignments ONCE. Every model reuses this exact list."""
    folds = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            folds.append({"seed": seed, "fold": k, "train": tr, "test": te})
    return folds


def check_no_group_leakage(folds: list[dict], groups: np.ndarray) -> None:
    """A hard assertion, not a comment. If this ever fires the results are void."""
    for f in folds:
        overlap = set(groups[f["train"]]) & set(groups[f["test"]])
        if overlap:
            raise AssertionError(
                f"subject(s) in both train and test (seed {f['seed']}, "
                f"fold {f['fold']}): {sorted(overlap)[:5]}")


def make_folds_leave_one_wave_out(y: np.ndarray, groups: np.ndarray,
                                  wave: np.ndarray) -> list[dict]:
    """Train on all waves but one, test on the held-out wave.

    A model riding wave-specific artifacts scores at chance on a wave it has
    never seen; one that learned physiology generalises. Waves whose test set is
    single-class are skipped and named -- AUC is undefined there, and the
    30-channel 2015/2016/2017 sessions are 100% uninjured.
    """
    folds, skipped = [], []
    for w in sorted({v for v in wave if v == v}):
        te = np.flatnonzero(wave == w)
        tr = np.flatnonzero(wave != w)
        if len(np.unique(y[te])) < 2 or len(te) < 20:
            skipped.append((w, len(te), int(y[te].sum())))
            continue
        # Grouping still applies: a subject seen in two waves must not straddle.
        overlap = set(groups[tr]) & set(groups[te])
        if overlap:
            tr = tr[~np.isin(groups[tr], list(overlap))]
        folds.append({"seed": int(w), "fold": 0, "train": tr, "test": te})
    if skipped:
        print("  LOWO skipped (single-class or too small): "
              + ", ".join(f"{w:.0f} (n={n}, pos={p})" for w, n, p in skipped))
    return folds


def build_preprocessor(numeric: list[str], categorical: list[str],
                       categorical_missing: str = "explicit",
                       pca_components: int | None = None) -> ColumnTransformer:
    """Median-impute numerics, one-hot categoricals. Fit inside the fold only.

    categorical_missing:
      "explicit"  -- keep 'missing' as its own level (what the data hands us)
      "mode"      -- impute to the most frequent level

    The spec requires both, because for `Level` an explicit missing level IS the
    provenance leak: a blank Level marks an uninjured session with 95% precision.
    The gap between the two fits is the size of that leak.
    """
    steps = []
    if numeric:
        num_steps = [("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())]
        if pca_components:
            # Waveforms are 101 correlated points per channel; raw they swamp the
            # sample size. PCA is fit INSIDE the fold like every other transform.
            num_steps.append(("pca", PCA(n_components=pca_components,
                                         random_state=0)))
        steps.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        cat_steps = []
        if categorical_missing == "mode":
            cat_steps.append(
                ("impute", SimpleImputer(strategy="most_frequent",
                                         missing_values="missing")))
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False)))
        steps.append(("cat", Pipeline(cat_steps), categorical))
    return ColumnTransformer(steps, remainder="drop")


def make_model(kind: str, numeric: list[str], categorical: list[str],
               categorical_missing: str = "explicit",
               pca_components: int | None = None):
    """Two families, per spec. Logistic gets inner-CV over C; HGB uses fixed
    defaults (stated in the report) rather than a grid, to keep the protocol from
    being tuned into the result."""
    pre = build_preprocessor(numeric, categorical, categorical_missing,
                             pca_components)
    if kind == "logit":
        clf = GridSearchCV(
            LogisticRegression(max_iter=5000, solver="lbfgs"),  # l2 is the default
            {"C": LOGIT_C_GRID},
            scoring="roc_auc",
            cv=StratifiedGroupKFold(n_splits=INNER_SPLITS),
            n_jobs=-1,
        )
        return Pipeline([("pre", pre), ("clf", clf)]), True
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(random_state=0)
        return Pipeline([("pre", pre), ("clf", clf)]), False
    raise ValueError(kind)


def evaluate(name: str, kind: str, frame: pd.DataFrame, numeric: list[str],
             categorical: list[str], folds: list[dict], groups: np.ndarray,
             categorical_missing: str = "explicit",
             pca_components: int | None = None) -> dict:
    """Score one model across every fold. Returns per-fold AUCs, aligned to `folds`
    so any two models can be differenced fold-by-fold."""
    y = frame["label"].to_numpy()
    X = frame[numeric + categorical]
    aucs, aps, chosen_c = [], [], []

    for f in folds:
        tr, te = f["train"], f["test"]
        model, needs_groups = make_model(kind, numeric, categorical,
                                         categorical_missing, pca_components)
        fit_kw = {}
        if needs_groups:
            fit_kw["clf__groups"] = groups[tr]
        model.fit(X.iloc[tr], y[tr], **fit_kw)
        prob = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        aps.append(average_precision_score(y[te], prob))
        if needs_groups:
            chosen_c.append(model.named_steps["clf"].best_params_["C"])

    aucs = np.asarray(aucs)
    return {
        "name": name,
        "kind": kind,
        "n_features": len(numeric) + len(categorical),
        "auc": aucs,
        "ap": np.asarray(aps),
        "auc_mean": float(aucs.mean()),
        "auc_ci": ci(aucs),
        "ap_mean": float(np.mean(aps)),
        "chosen_C": chosen_c,
    }


def ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval across folds. Not a bootstrap -- these are the actual
    per-fold estimates, which is what CLAUDE.md asks to be reported."""
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_delta(a: dict, b: dict) -> dict:
    """a - b, fold by fold. Requires both scored on the same fold list."""
    if a["auc"].shape != b["auc"].shape:
        raise ValueError("models were not scored on the same folds")
    d = a["auc"] - b["auc"]
    lo, hi = ci(d)
    return {
        "a": a["name"], "b": b["name"],
        "delta_mean": float(d.mean()),
        "delta_ci": (lo, hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "wins": int((d > 0).sum()),
        "n": int(len(d)),
    }


def fmt_result(r: dict) -> str:
    lo, hi = r["auc_ci"]
    return (f"{r['name']:<38} AUC {r['auc_mean']:.3f} [{lo:.3f}, {hi:.3f}]  "
            f"PR-AUC {r['ap_mean']:.3f}  ({r['n_features']} feat)")


def fmt_delta(d: dict) -> str:
    lo, hi = d["delta_ci"]
    verdict = "EXCLUDES ZERO" if d["excludes_zero"] else "includes zero"
    return (f"{d['a']} - {d['b']}: dAUC {d['delta_mean']:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}]  {verdict}  "
            f"({d['wins']}/{d['n']} folds positive)")
