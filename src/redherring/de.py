"""Distracting effect [3]:  DE(q, p) = 1 - P(model abstains | q, p alone),  for p without the answer.

Thinking off: P(abstain) from the logprobs of the abstain string (exact).
Thinking on:  P(abstain) estimated as the fraction of SAMPLES sampled answers that abstain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import ABSTAIN, de_messages
from .utils import log, read_jsonl, run_dir, write_jsonl

if TYPE_CHECKING:  # rq2 imports this module and shouldn't need vllm installed
    from .llm import Model

SAMPLES = 8


def load_retrieval(cfg: dict) -> list[dict]:
    """Retrieval rows plus `relevant` (top passage containing the answer, or None) and
    `candidates` (top passages NOT containing the answer = potential distractors)."""
    rows = list(read_jsonl(run_dir(cfg) / "retrieval.jsonl"))
    for r in rows:
        r["relevant"] = next((p for p in r["passages"] if p["has_answer"]), None)
        r["candidates"] = [p for p in r["passages"] if not p["has_answer"]][:cfg["de"]["candidates"]]
    return rows


def score_de(cfg: dict, model: Model, mode: str):
    pairs = [(r, p) for r in load_retrieval(cfg) for p in r["candidates"]]
    msgs = [de_messages(r["question"], p) for r, p in pairs]
    budget = cfg["modes"][mode]
    log.info("DE: %d pairs, mode=%s", len(pairs), mode)
    if budget == 0:
        des = [1.0 - pa for pa in model.abstain_prob(msgs)]
    else:
        gens = model.generate(msgs, budget, n=SAMPLES)
        des = [sum(ABSTAIN.lower() not in g.answer.lower() for g in row) / len(row) for row in gens]
    return write_jsonl(run_dir(cfg) / f"de_{mode}.jsonl",
                       ({"qid": r["qid"], "pid": p["pid"], "de": d} for (r, p), d in zip(pairs, des)))


def load_de(cfg: dict, mode: str) -> dict[tuple[str, int], float]:
    return {(r["qid"], r["pid"]): r["de"] for r in read_jsonl(run_dir(cfg) / f"de_{mode}.jsonl")}
