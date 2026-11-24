"""PopQA + Wikipedia corpus: download and loading."""

from __future__ import annotations

import csv
import gzip
import json
import random
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import requests
from tqdm import tqdm

from .utils import DATA, log

csv.field_size_limit(sys.maxsize)

POPQA_URL = "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv"
# DPR Dec-2018 Wikipedia: 21M 100-word passages (id, text, title).  TODO: confirm this is [4]'s corpus.
CORPUS_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
CORPUS_SIZE = 21_015_324

POPQA = DATA / "popqa.tsv"
CORPUS = DATA / "psgs_w100.tsv"


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        log.info("already have %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f, tqdm(total=int(r.headers.get("content-length", 0)) or None,
                                        unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(1 << 20):
                bar.update(f.write(block))
    tmp.replace(dest)
    return dest


def setup(models: list[str]) -> None:
    """Download PopQA, the corpus, and model weights (into $HF_HOME)."""
    download(POPQA_URL, POPQA)
    if not CORPUS.exists():
        gz = download(CORPUS_URL, CORPUS.with_name(CORPUS.name + ".gz"))
        log.info("decompressing %s", gz)
        with gzip.open(gz, "rb") as fi, open(CORPUS.with_suffix(".tmp"), "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 24)
        CORPUS.with_suffix(".tmp").replace(CORPUS)
        gz.unlink()
    from huggingface_hub import snapshot_download
    for m in models:
        log.info("fetching %s", m)
        snapshot_download(m)


def load_popqa(cfg: dict) -> list[dict]:
    """A fixed random sample of n_questions PopQA rows."""
    rows = []
    with open(POPQA, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append({"qid": f"popqa-{r['id']}", "question": r["question"],
                         "answers": json.loads(r["possible_answers"]), "subj": r["subj"],
                         "s_aliases": json.loads(r["s_aliases"] or "[]"), "prop": r["prop"],
                         "s_pop": int(r["s_pop"])})
    return random.Random(cfg["seed"]).sample(rows, min(cfg["n_questions"], len(rows)))


def iter_corpus(start: int = 0, end: int | None = None) -> Iterator[dict]:
    """Corpus rows with index in [start, end)."""
    with open(CORPUS, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            yield {"pid": int(row[0]), "text": row[1], "title": row[2]}


def fetch_passages(pids: set[int]) -> dict[int, dict]:
    """One pass over the corpus to pull specific passages (avoids holding 21M in RAM)."""
    found = {}
    for p in iter_corpus():
        if p["pid"] in pids:
            found[p["pid"]] = p
            if len(found) == len(pids):
                break
    return found
