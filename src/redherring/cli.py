"""rh [-c config] [-o key=value ...] <stage>

  setup      download PopQA, Wikipedia corpus, LLM + retriever weights
  encode     embed the corpus (resumable)
  retrieve   top-k passages per question            -> retrieval.jsonl
  de         DE of candidate distractors per mode   -> de_<mode>.jsonl
  rq1        accuracy: modes x mixtures             -> preds_<mode>.jsonl, rq1_summary.json
  rq2        hard-vs-weak classifier + DE rank corr -> rq2_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging

from .utils import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="rh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="configs/default.yaml")
    p.add_argument("-o", "--override", action="append", default=[], help="key.path=value, repeatable")
    p.add_argument("stage", choices=["setup", "encode", "retrieve", "de", "rq1", "rq2"])
    p.add_argument("--modes", nargs="+", default=None, help="subset of config modes (default: all)")
    p.add_argument("--limit", type=int, default=None, help="rq1: only the first N questions (smoke test)")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config, a.override)
    modes = a.modes or list(cfg["modes"])

    # per-stage imports: only load torch/vllm for the stages that need them
    if a.stage == "setup":
        from .data import setup
        from .retrieval import RETRIEVER
        setup([cfg["model"], RETRIEVER])
    elif a.stage == "encode":
        from .retrieval import encode_corpus
        encode_corpus()
    elif a.stage == "retrieve":
        from .retrieval import retrieve
        print(retrieve(cfg))
    elif a.stage == "de":
        from .de import score_de
        from .llm import Model
        model = Model(cfg["model"], cfg["seed"])
        for mode in modes:
            print(score_de(cfg, model, mode))
    elif a.stage == "rq1":
        from .llm import Model
        from .rq1 import run_rq1
        s = run_rq1(cfg, Model(cfg["model"], cfg["seed"]), modes, a.limit)
        for k, v in s["accuracy"].items():
            print(f"{k:28s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]  n={v['n']}")
    elif a.stage == "rq2":
        from .rq2 import run_rq2
        print(json.dumps(run_rq2(cfg, modes), indent=2))


if __name__ == "__main__":
    main()
