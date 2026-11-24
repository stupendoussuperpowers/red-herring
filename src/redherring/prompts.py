"""Prompt templates. Messages are OpenAI-style [{role, content}] lists; the backend applies the chat template."""

from __future__ import annotations

ABSTAIN = "NO-RESPONSE"

QA_INSTRUCTION = (
    "Answer the question using the documents below. "
    "Reply with only the short answer (a few words), no explanation."
)

# Verbatim from [3] (Amiraz et al., Fig. 6).
DE_INSTRUCTION = (
    "You are given a question and you must respond based on the provided documents. "
    "Respond directly without providing any premise or explanation. "
    f"If none of the documents contain the answer, please respond with {ABSTAIN}. "
    "Do not try to respond based on your own knowledge."
)

JUDGE_INSTRUCTION = (
    "You are grading a question-answering system. Given the question, the gold answers, and the "
    "system's prediction, decide whether the prediction is correct (it refers to the same entity or "
    "value as a gold answer; extra wording is fine). Reply with exactly 'yes' or 'no'."
)


def format_passage(i: int, p: dict) -> str:
    return f"Document [{i + 1}] (Title: {p.get('title', '')}) {p['text']}"


def qa_messages(question: str, passages: list[dict]) -> list[dict]:
    docs = "\n".join(format_passage(i, p) for i, p in enumerate(passages))
    body = f"{QA_INSTRUCTION}\n\n{docs}\n\nQuestion: {question}" if passages else \
        f"Reply with only the short answer (a few words), no explanation.\n\nQuestion: {question}"
    return [{"role": "user", "content": body}]


def de_messages(question: str, passage: dict) -> list[dict]:
    return [{"role": "user", "content": (
        f"{DE_INSTRUCTION}\nDocuments: {format_passage(0, passage)}\nQuestion: {question}\nAnswer:")}]


def judge_messages(question: str, golds: list[str], pred: str) -> list[dict]:
    return [{"role": "user", "content": (
        f"{JUDGE_INSTRUCTION}\n\nQuestion: {question}\nGold answers: {' | '.join(golds[:10])}\n"
        f"Prediction: {pred}\n\nCorrect?"
    )}]
