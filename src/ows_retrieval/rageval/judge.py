"""Build the judge RAGAS scores with: a (llm, embeddings) pair it can call.

RAGAS does not score with a formula -- it prompts a judge LLM (faithfulness:
extract then verify claims; answer relevancy: generate questions from the answer)
and uses an embedder (answer relevancy: compare those questions to the real one).
This builds that pair, each wrapped in the RAGAS adapter so the metrics can call it.

Both run locally via Ollama. Keeping it in one function means the smoke test and
the full run configure the judge identically and the model choice lives in one
place. temperature=0 keeps the judge as deterministic as Ollama allows, so
re-running the scoring reproduces the same numbers (matching the generation config).
"""
# Import pyarrow before langchain_ollama: on Windows, importing langchain_ollama
# first makes pyarrow's bundled DLLs crash on load (access violation). pyarrow
# arrives transitively via ragas -> datasets -> pandas, so force it up front.
import pyarrow  # noqa: F401

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

JUDGE_MODEL = "gemma4:e4b"        # reasoning model, already used for generation
EMBED_MODEL = "nomic-embed-text"  # dedicated embedder for answer relevancy


def make_judge(judge_model: str = JUDGE_MODEL, embed_model: str = EMBED_MODEL):
    """Return ``(llm, embeddings)`` wrapped for RAGAS, both backed by local Ollama.

    num_ctx=16384 gives the judge room for the largest k=50 prompts: the default
    Ollama context (~2048) truncates them, so the judge sees garbled input and emits
    output RAGAS cannot parse. reasoning=False disables gemma4's hidden thinking (the
    same `think: false` the generation needed), keeping the output clean for the parser.
    """
    llm = LangchainLLMWrapper(
        ChatOllama(model=judge_model, temperature=0, reasoning=False, num_ctx=16384)
    )
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=embed_model))
    return llm, embeddings


if __name__ == "__main__":
    llm, embeddings = make_judge()
    print(f"judge LLM:   {type(llm).__name__} -> {JUDGE_MODEL}")
    print(f"embeddings:  {type(embeddings).__name__} -> {EMBED_MODEL}")
    print("both constructed OK")
