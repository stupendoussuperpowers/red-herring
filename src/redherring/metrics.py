"""Answer matching and bootstrap confidence intervals."""

from __future__ import annotations

import re
import string
import unicodedata

import numpy as np


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization, plus unicode NFKC and handling of non-Latin text."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) and not unicodedata.category(ch).startswith("P"))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def contains_match(text: str, golds: list[str]) -> bool:
    """True if any normalized gold answer is in the normalized text (as whole words)."""
    t = f" {normalize_answer(text)} "
    return any((g := normalize_answer(x)) and f" {g} " in t for x in golds)


def _boot_means(v: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)


def bootstrap_ci(values, n: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean and percentile bootstrap CI over examples."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    means = _boot_means(v, n, seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def paired_bootstrap(a, b, n: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI for mean(b - a) over the same examples (e.g. thinking vs no thinking).

    Also returns a one-sided bootstrap p-value for "b is not better than a".
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired bootstrap needs aligned examples"
    out = bootstrap_ci(b - a, n=n, alpha=alpha, seed=seed)
    out["p_le_0"] = float((_boot_means(b - a, n, seed) <= 0).mean()) if a.size else None
    return out
