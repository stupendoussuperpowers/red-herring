"""RQ2: which passage features predict hard (DE > hard_min) vs weak (DE < weak_max) distractors,
and do DE rankings agree across thinking modes?

1. features: cheap passage-level features for every (question, candidate). Add new ones here.
2. per mode: logistic regression hard vs weak, held-out AUC (train/test split by question),
   coefficients, and per-feature ablation (AUC lost when the feature is dropped).
3. Spearman correlation of DE between every pair of modes (global and mean within-question).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .de import load_de, load_retrieval
from .metrics import contains_match, normalize_answer
from .utils import log, run_dir

STOP = set("a an the of in on at to for is was what who which where when how by and or with from as".split())
FEATURES = ["retrieval_score", "rank", "bm25", "q_overlap", "subj_in_passage", "subj_in_title",
            "answer_type_present", "log_len", "log_s_pop"]


def _toks(s: str) -> list[str]:
    return [t for t in re.findall(r"\w+", s.lower()) if t not in STOP]


def _bm25(query: list[str], doc: list[str], df: Counter, n_docs: int, avgdl: float, k1=0.9, b=0.4) -> float:
    tf = Counter(doc)
    return sum(math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * (k1 + 1)
               / (tf[t] + k1 * (1 - b + b * len(doc) / avgdl)) for t in set(query) if t in tf)


def features(rows: list[dict]) -> tuple[list[tuple[str, int]], np.ndarray]:
    """(keys, X) for every (question, candidate distractor); columns = FEATURES."""
    docs = {(r["qid"], p["pid"]): _toks(p["title"] + " " + p["text"]) for r in rows for p in r["candidates"]}
    df = Counter(t for d in docs.values() for t in set(d))  # BM25 stats over the candidate pool
    avgdl = float(np.mean([len(d) for d in docs.values()]))
    # Answer-type proxy: passage mentions a gold answer of *another* question with the same relation.
    by_prop = defaultdict(set)
    for r in rows:
        by_prop[r["prop"]].update(normalize_answer(a) for a in r["answers"][:3])

    keys, X = [], []
    for r in rows:
        q = _toks(r["question"])
        same_type = list(by_prop[r["prop"]] - {normalize_answer(a) for a in r["answers"]})
        subj = [r["subj"], *r["s_aliases"]]
        for p in r["candidates"]:
            d = docs[(r["qid"], p["pid"])]
            keys.append((r["qid"], p["pid"]))
            X.append([p["score"], p["rank"], _bm25(q, d, df, len(docs), avgdl),
                      len(set(q) & set(d)) / max(len(set(q)), 1),
                      float(contains_match(p["text"], subj)), float(contains_match(p["title"], subj)),
                      float(contains_match(p["text"], same_type)), math.log(1 + len(d)), math.log(1 + r["s_pop"])])
    return keys, np.asarray(X, dtype=float)


def _fit(X, y, groups, seed):
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])), clf


def run_rq2(cfg: dict, modes: list[str]) -> dict:
    keys, X = features(load_retrieval(cfg))
    groups = np.array([q for q, _ in keys])
    lo, hi, seed = cfg["de"]["weak_max"], cfg["de"]["hard_min"], cfg["seed"]
    des = {m: load_de(cfg, m) for m in modes}
    out = {"per_mode": {}, "spearman": {}}

    for m, de in des.items():
        d = np.array([de[k] for k in keys])
        mask = (d > hi) | (d < lo)
        y = (d[mask] > hi).astype(int)
        res = {"n": int(mask.sum()), "n_hard": int(y.sum())}
        if len(set(y)) < 2:
            log.warning("mode %s: only one class present, skipping classifier", m)
        else:
            res["auc"], clf = _fit(X[mask], y, groups[mask], seed)
            res["coef"] = dict(zip(FEATURES, clf[-1].coef_[0].round(4).tolist()))
            res["ablation_auc_drop"] = {
                f: round(res["auc"] - _fit(np.delete(X[mask], i, 1), y, groups[mask], seed)[0], 4)
                for i, f in enumerate(FEATURES)}
        out["per_mode"][m] = res

    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            va, vb = [des[a][k] for k in keys], [des[b][k] for k in keys]
            per_q = defaultdict(lambda: ([], []))
            for k, x, y in zip(keys, va, vb):
                per_q[k[0]][0].append(x)
                per_q[k[0]][1].append(y)
            within = [spearmanr(x, y).statistic for x, y in per_q.values() if len(set(x)) > 1 and len(set(y)) > 1]
            out["spearman"][f"{a}~{b}"] = {"global": float(spearmanr(va, vb).statistic),
                                           "mean_within_question": float(np.mean(within)) if within else None}

    (run_dir(cfg) / "rq2_summary.json").write_text(json.dumps(out, indent=2))
    return out
