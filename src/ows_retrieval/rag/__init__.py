"""RAG pipeline: build prompts from retrieved documents, generate answers with a
local Gemma model, and cache them.

Modules:
    context   -- resolve retrieved doc_ids into the text fed to the generator
    prompt    -- wrap a query and its context block in the fixed template
    generator -- send one prompt to local Gemma (Ollama) and return the answer
    cache     -- read/write the per-answer JSON cache (one file per key)
"""
