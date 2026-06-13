"""Generate one answer from one prompt using a local Gemma model via Ollama.

A thin wrapper around the generator: it turns a prepared prompt into an answer.
It knows nothing about caching, run files, or the k-sweep -- the sweep driver
decides which prompts to send and what to do with the answers. Uses stdlib
`urllib` only (no extra dependency); Ollama must be running locally.
"""
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class GenerationResult:
    """One Gemma answer plus the token counts that flag silent truncation.

    prompt_tokens (Ollama's prompt_eval_count): if it ever exceeds
        num_ctx - num_predict, the context was silently truncated -> answer suspect.
    output_tokens (Ollama's eval_count): if it equals num_predict, the answer
        likely hit the length cap and was cut off -> quality flag for M3.
    """
    text: str
    prompt_tokens: int
    output_tokens: int


def gemma_generate(
    prompt: str,
    model: str,
    options: dict,
    timeout: float = 300.0,
    think: bool | None = None,
) -> GenerationResult:
    """Send one prompt to Ollama and return the answer with its token counts.

    `options` is passed straight through to Ollama (temperature, seed, num_ctx,
    num_predict). `think` controls Ollama's top-level reasoning channel: leave it
    None for non-reasoning models (e.g. gemma2) so the field is omitted, or set
    False for a reasoning model (e.g. gemma4) so it answers directly instead of
    spending num_predict on hidden reasoning (which yields an empty `response`).

    Raises RuntimeError if the server is unreachable or the response has no answer
    text, so a failed call never silently caches an empty answer.
    """
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    if think is not None:
        payload["think"] = think
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as err:
        raise RuntimeError(f"Ollama request failed ({OLLAMA_URL}): {err}") from err

    answer = body.get("response", "").strip()
    if not answer:
        raise RuntimeError(f"Ollama returned no answer text: {body}")

    return GenerationResult(
        text=answer,
        prompt_tokens=body["prompt_eval_count"],
        output_tokens=body["eval_count"],
    )
