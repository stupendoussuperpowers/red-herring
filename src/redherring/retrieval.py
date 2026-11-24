"""Dense retrieval: corpus encoding (checkpointed in shards) + exact inner-product search.

Search streams the fp16 shards through the GPU and keeps a running top-k: no FAISS needed,
and the full 21M x d matrix never sits in RAM at once.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .data import CORPUS_SIZE, fetch_passages, iter_corpus, load_popqa
from .metrics import contains_match
from .utils import DATA, log, run_dir, write_jsonl

# TODO: confirm this is the dense retriever used in [4].
RETRIEVER = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 100
N_SHARDS = 16
INDEX = DATA / "index" / RETRIEVER.replace("/", "__")


def _encoder() -> SentenceTransformer:
    return SentenceTransformer(RETRIEVER, device="cuda", model_kwargs={"torch_dtype": torch.float16})


def _encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, batch_size=512, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)


def encode_corpus() -> None:
    """Embed the corpus into N_SHARDS files. Finished shards are skipped, so re-running resumes."""
    INDEX.mkdir(parents=True, exist_ok=True)
    model = None
    for s in range(N_SHARDS):
        emb_path = INDEX / f"emb_{s:03d}.npy"
        if emb_path.exists():
            continue
        model = model or _encoder()
        start, end = s * CORPUS_SIZE // N_SHARDS, (s + 1) * CORPUS_SIZE // N_SHARDS
        log.info("encoding shard %d/%d [%d, %d)", s + 1, N_SHARDS, start, end)
        embs, ids, buf = [], [], []
        for p in tqdm(iter_corpus(start, end), total=end - start, mininterval=30):
            buf.append(f"{p['title']}\n{p['text']}")
            ids.append(p["pid"])
            if len(buf) == 8192:
                embs.append(_encode(model, buf).astype(np.float16)); buf = []
        if buf:
            embs.append(_encode(model, buf).astype(np.float16))
        np.save(INDEX / f"ids_{s:03d}.npy", np.asarray(ids, dtype=np.int64))
        np.save(INDEX / "tmp.npy", np.concatenate(embs))
        (INDEX / "tmp.npy").replace(emb_path)  # emb file appears last = shard complete


def search(queries: list[str], k: int = TOP_K, block: int = 262144) -> tuple[np.ndarray, np.ndarray]:
    """(scores [Q,k], pids [Q,k]), best first."""
    q = torch.from_numpy(_encode(_encoder(), [QUERY_PREFIX + x for x in queries])).cuda().float()
    best_s = torch.empty((len(queries), 0), device="cuda")
    best_i = torch.empty((len(queries), 0), dtype=torch.long, device="cuda")
    for s in tqdm(range(N_SHARDS), desc="search"):
        emb = np.load(INDEX / f"emb_{s:03d}.npy", mmap_mode="r")
        ids = torch.from_numpy(np.load(INDEX / f"ids_{s:03d}.npy")).cuda()
        for a in range(0, len(emb), block):
            chunk = torch.from_numpy(np.ascontiguousarray(emb[a:a + block])).cuda().float()
            ts, ti = torch.topk(q @ chunk.T, min(k, len(chunk)))
            best_s, order = torch.topk(torch.cat([best_s, ts], 1), min(k, best_s.shape[1] + ts.shape[1]))
            best_i = torch.gather(torch.cat([best_i, ids[a + ti]], 1), 1, order)
    return best_s.cpu().numpy(), best_i.cpu().numpy()


def retrieve(cfg: dict):
    """Top-k passages for each PopQA question, flagged with whether they contain a gold answer."""
    qs = load_popqa(cfg)
    scores, pids = search([q["question"] for q in qs])
    psg = fetch_passages({int(p) for p in pids.ravel()})
    for q, ss, ps in zip(qs, scores, pids):
        q["passages"] = [{**psg[int(p)], "score": float(s), "rank": r,
                          "has_answer": contains_match(psg[int(p)]["text"], q["answers"])}
                         for r, (s, p) in enumerate(zip(ss, ps))]
    return write_jsonl(run_dir(cfg) / "retrieval.jsonl", qs)
