"""Count judge-LLM token usage during a RAGAS run, via a langchain callback.

RAGAS makes its judge calls through langchain, so a callback handler sees every
LLM response and can sum the token usage langchain reports. We read the standard
``usage_metadata`` (input/output tokens) and fall back to Ollama's raw counts
(``prompt_eval_count`` / ``eval_count``) if it is absent. Embedding calls are not
LLM calls and are not counted.

Local Ollama tokens are free, but this is the meter that matters the moment a
hosted (paid, rate-limited) judge is swapped in.
"""
from langchain_core.callbacks import BaseCallbackHandler


class TokenCounter(BaseCallbackHandler):
    """Accumulate input/output token counts across all LLM calls in a run."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for gen in generations:
                self.llm_calls += 1
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) if message is not None else None
                if usage:
                    self.input_tokens += usage.get("input_tokens") or 0
                    self.output_tokens += usage.get("output_tokens") or 0
                elif message is not None:
                    meta = getattr(message, "response_metadata", {}) or {}
                    self.input_tokens += meta.get("prompt_eval_count") or 0
                    self.output_tokens += meta.get("eval_count") or 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
