from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

log = logging.getLogger("redherring")

DATA = Path("data")       # datasets, corpus, index (~13GB corpus + ~40GB index)
OUTPUTS = Path("outputs")


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """YAML config + `a.b=value` overrides (values parsed as YAML)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides:
        key, _, raw = ov.partition("=")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def run_dir(cfg: dict) -> Path:
    d = OUTPUTS / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Append and fsync, so rows written before a crash survive it."""
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint(path: str | Path) -> list[dict]:
    """Rows from an append-only file. A half-written last line (crash mid-write) is cut off."""
    path = Path(path)
    if not path.exists():
        return []
    rows, good = [], 0
    with open(path, "rb") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
            good += len(line)
    if good < path.stat().st_size:
        log.warning("%s: dropping a partially written line", path)
        with open(path, "r+b") as f:
            f.truncate(good)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    """Write via a tmp file so a killed run never leaves a half-written file behind."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path
