"""RQ1: accuracy of each thinking mode across distractor mixtures.

1. build_prompts: for each question, 1 relevant passage + each mixture's hard/weak distractors
   (labels from no_think DE). Only questions that can fill *every* mixture are kept, and mixtures
   are seeded by (qid, mixture), so every mode/mixture is compared on identical inputs (paired).
2. run_rq1: generate under each mode, mark correct = string match OR LLM judge. Resumable:
   results are appended every CHUNK prompts and finished prompts are skipped on re-run.
3. summarize: accuracy with 95% bootstrap CI, paired deltas vs no_think and vs gold_only.
"""

from __future__ import annotations

import json
import random
import zlib
from collections import defaultdict

from .de import load_de, load_retrieval
from .llm import Model
from .metrics import bootstrap_ci, contains_match, paired_bootstrap
from .prompts import judge_messages, qa_messages
from .utils import append_jsonl, load_checkpoint, log, read_jsonl, run_dir

CHUNK = 500  # prompts per checkpoint


def build_prompts(cfg: dict) -> list[dict]:
    de = load_de(cfg, "no_think")
    mixes, lo, hi = cfg["mixtures"], cfg["de"]["weak_max"], cfg["de"]["hard_min"]
    need_h = max(m["hard"] for m in mixes.values())
    need_w = max(m["weak"] for m in mixes.values())
    rows, out, kept = load_retrieval(cfg), [], 0
    for r in rows:
        hard = [p for p in r["candidates"] if de[(r["qid"], p["pid"])] > hi]
        weak = [p for p in r["candidates"] if de[(r["qid"], p["pid"])] < lo]
        if r["relevant"] is None or len(hard) < need_h or len(weak) < need_w:
            continue
        kept += 1
        for name, m in mixes.items():
            rng = random.Random(zlib.crc32(f"{cfg['seed']}|{r['qid']}|{name}".encode()))
            psgs = [dict(r["relevant"], role="relevant")] + \
                   [dict(p, role="hard") for p in rng.sample(hard, m["hard"])] + \
                   [dict(p, role="weak") for p in rng.sample(weak, m["weak"])]
            rng.shuffle(psgs)
            out.append({"qid": r["qid"], "question": r["question"], "answers": r["answers"], "mixture": name,
                        "relevant_pos": next(i for i, p in enumerate(psgs) if p["role"] == "relevant"),
                        "passages": psgs})
    log.info("RQ1: %d/%d questions usable for all mixtures", kept, len(rows))
    return out


def run_rq1(cfg: dict, model: Model, modes: list[str], limit: int | None = None) -> dict:
    recs = build_prompts(cfg)
    if limit:
        keep = set(sorted({r["qid"] for r in recs})[:limit])
        recs = [r for r in recs if r["qid"] in keep]
    for r in recs:
        r["pids"] = [p["pid"] for p in r["passages"]]

    # Checkpointing: preds_<mode>.jsonl is append-only and written every CHUNK prompts. On restart,
    # prompts already in it are skipped. The key includes the passage ids, so if the mixtures change
    # (new seed/config) stale rows are not reused; use a new run_name in that case anyway.
    def key(r):
        return (r["qid"], r["mixture"], tuple(r["pids"]))

    for mode in modes:
        path = run_dir(cfg) / f"preds_{mode}.jsonl"
        done = {key(r) for r in load_checkpoint(path)}
        todo = [r for r in recs if key(r) not in done]
        log.info("RQ1: mode=%s, %d prompts (%d already done)", mode, len(recs), len(recs) - len(todo))
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            gens = [g[0] for g in model.generate([qa_messages(r["question"], r["passages"]) for r in chunk],
                                                 cfg["modes"][mode])]
            jm = [judge_messages(r["question"], r["answers"], g.answer) for r, g in zip(chunk, gens)]
            judged = [g[0].answer.lower().startswith("yes") for g in model.generate(jm, 0, greedy=True, max_tokens=4)]
            append_jsonl(path, (
                {"qid": r["qid"], "mixture": r["mixture"], "pids": r["pids"], "relevant_pos": r["relevant_pos"],
                 "pred": g.answer, "thinking": g.thinking, "think_tokens": g.think_tokens,
                 "truncated_thinking": g.truncated_thinking,
                 "match": (cm := contains_match(g.answer, r["answers"])), "judge": j, "correct": cm or j}
                for r, g, j in zip(chunk, gens, judged)))
            log.info("RQ1: mode=%s, %d/%d new prompts done", mode, i + len(chunk), len(todo))

    summary = summarize(cfg, modes)
    (run_dir(cfg) / "rq1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize(cfg: dict, modes: list[str]) -> dict:
    acc = defaultdict(dict)  # (mode, mixture) -> {qid: 0/1}
    for mode in modes:
        for r in read_jsonl(run_dir(cfg) / f"preds_{mode}.jsonl"):
            acc[(mode, r["mixture"])][r["qid"]] = float(r["correct"])

    out = {"accuracy": {}, "delta_vs_no_think": {}, "delta_vs_gold_only": {}}
    for (mode, mix), d in acc.items():
        qids = sorted(d)
        vals = [d[q] for q in qids]
        out["accuracy"][f"{mode}/{mix}"] = bootstrap_ci(vals)
        for key, base in (("delta_vs_no_think", ("no_think", mix)), ("delta_vs_gold_only", (mode, "gold_only"))):
            if base != (mode, mix) and base in acc:
                out[key][f"{mode}/{mix}"] = paired_bootstrap([acc[base][q] for q in qids], vals)
    return out
