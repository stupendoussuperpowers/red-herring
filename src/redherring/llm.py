"""vLLM wrapper with Qwen3 thinking control. A mode is just a thinking budget:

  0     -> thinking off (chat template with enable_thinking=False)
  None  -> uncapped thinking
  B     -> budget forcing: think up to B tokens (stop at </think>), close the think block, then answer
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from .prompts import ABSTAIN

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
EARLY_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
ANSWER_TOKENS = 64        # short-answer QA
MAX_MODEL_LEN = 32768
UNCAPPED_TOKENS = 28000   # thinking + answer when the budget is None


@dataclass
class Generation:
    answer: str
    thinking: str = ""
    think_tokens: int = 0
    truncated_thinking: bool = False


def split_thinking(raw: str) -> tuple[str, str]:
    """'<think>...</think>answer' -> (thinking, answer). No closing tag => everything is thinking."""
    if THINK_CLOSE not in raw:
        return raw.replace(THINK_OPEN, "").strip(), ""
    think, ans = raw.split(THINK_CLOSE, 1)
    return think.replace(THINK_OPEN, "").strip(), ans.strip()


class Model:
    def __init__(self, name: str, seed: int):
        self.llm = LLM(model=name, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9,
                       seed=seed, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        self.seed = seed

    def _template(self, msgs: list[dict], thinking: bool) -> str:
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=thinking)

    def _params(self, thinking: bool, max_tokens: int, n: int = 1, greedy: bool = False, **kw) -> SamplingParams:
        # Qwen3's recommended sampling settings for each mode
        t, p = (0.6, 0.95) if thinking else (0.7, 0.8)
        if greedy:
            t, p = 0.0, 1.0
        return SamplingParams(n=n, temperature=t, top_p=p, top_k=20, max_tokens=max_tokens, seed=self.seed, **kw)

    def generate(self, messages: list[list[dict]], budget: int | None, n: int = 1,
                 greedy: bool = False, max_tokens: int = ANSWER_TOKENS) -> list[list[Generation]]:
        """n generations per prompt."""
        if budget == 0:
            outs = self.llm.generate([self._template(m, False) for m in messages],
                                     self._params(False, max_tokens, n, greedy))
            return [[Generation(c.text.strip()) for c in o.outputs] for o in outs]

        prompts = [self._template(m, True) for m in messages]
        if budget is None:
            outs = self.llm.generate(prompts, self._params(True, UNCAPPED_TOKENS, n))
            res = []
            for o in outs:
                row = []
                for c in o.outputs:
                    think, ans = split_thinking(c.text)
                    row.append(Generation(ans, think, len(self.tok.encode(think)), THINK_CLOSE not in c.text))
                res.append(row)
            return res

        # Stage 1: think, capped at `budget` tokens.
        stage1 = [p if p.rstrip().endswith(THINK_OPEN) else p + THINK_OPEN + "\n" for p in prompts]
        outs1 = self.llm.generate(stage1, self._params(True, budget, n, stop=[THINK_CLOSE]))
        stage2, meta = [], []
        for p, o in zip(stage1, outs1):
            for c in o.outputs:
                truncated = c.finish_reason == "length"
                stage2.append(p + c.text.rstrip() + (EARLY_STOP if truncated else "") + f"\n{THINK_CLOSE}\n\n")
                meta.append((c.text.strip(), len(c.token_ids), truncated))
        # Stage 2: answer with the closed think block in context.
        outs2 = self.llm.generate(stage2, self._params(True, max_tokens))
        gens = [Generation(o.outputs[0].text.strip(), *m) for o, m in zip(outs2, meta)]
        return [gens[i * n:(i + 1) * n] for i in range(len(messages))]

    def abstain_prob(self, messages: list[list[dict]]) -> list[float]:
        """P(first token of the reply == first token of ABSTAIN), thinking off. As in [3] (Alg. 1)."""
        first = self.tok.encode(ABSTAIN, add_special_tokens=False)[0]
        prompts, starts = [], []
        for m in messages:
            ids = self.tok.encode(self._template(m, False), add_special_tokens=False)
            starts.append(len(ids))
            prompts.append(TokensPrompt(prompt_token_ids=ids + [first]))
        outs = self.llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0))
        return [math.exp(o.prompt_logprobs[s][first].logprob) for o, s in zip(outs, starts)]
