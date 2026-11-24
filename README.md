# red-herring

Controlled distractor studies in RAG (see `proposal.pdf`).

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[gpu]"
export HF_HOME=/path/with/space/hf   # model cache

rh setup              # PopQA, DPR Wikipedia (~13GB), Qwen3-8B, retriever weights
rh encode             # embed 21M passages; resumable, re-run if it dies
rh retrieve           # -> outputs/popqa/retrieval.jsonl
```

## Running

```bash
rh de --modes no_think think_512 think_full   # -> de_<mode>.jsonl
rh rq1 --limit 20     # smoke test, then without --limit
rh rq2
```

## Outputs

| file                 | contents                                                                                                    |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `retrieval.jsonl`    | question, answers, top-k passages with `has_answer`                                                         |
| `de_<mode>.jsonl`    | DE per (question, candidate distractor)                                                                     |
| `preds_<mode>.jsonl` | prediction, thinking trace, string match, judge verdict, per (question, mixture)                            |
| `rq1_summary.json`   | accuracy with 95% bootstrap CIs; paired deltas vs `no_think` and vs `gold_only`                             |
| `rq2_summary.json`   | hard vs weak logistic regression AUC (split by question), coefficients, ablations, DE Spearman across modes |

## Notes

- **DE** follows [3]: `1 - P(NO-RESPONSE | q, p)`, using [3]'s prompt. With thinking off it is the
  first-token probability (as in [3]); with thinking on it is estimated from 8 samples.
- **think_512** uses budget forcing: think for up to 512 tokens, then close `</think>` and answer.
- **Mixtures:** each prompt has the top retrieved passage containing the answer plus the configured hard and
  weak distractors in shuffled order. A question is used only if it can fill every mixture, so all
  comparisons are paired.
- **TODO:** check that the corpus (`data.py: CORPUS_URL`) and retriever (`retrieval.py: RETRIEVER`) match [4].
