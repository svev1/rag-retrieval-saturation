"""Build the fixed prompt that wraps a query and its retrieved context.

Every cached answer in the k-sweep is generated from a prompt produced here. The
template is frozen on purpose: only the retriever and k are allowed to vary across
the cache, so any answer-quality difference is attributable to them and not to
the wording. If the template ever changes, bump `PROMPT_TEMPLATE_VERSION` so
mixed-version answers can be spotted in the provenance.
"""

# Bump this if the template string below changes -- even by a single character.
# Stamped into every cached answer's provenance so M3 can detect mixed templates.
PROMPT_TEMPLATE_VERSION = 1

PROMPT_TEMPLATE = """\
You are writing a brief answer to a search query. Use ONLY information from the \
documents below; do not use outside knowledge. If the documents do not contain \
the answer, reply exactly: "The documents do not answer this." Refer to documents \
inline as [Document N]. Keep the answer to 2-4 sentences.

Query: {query}

Documents:
{context_block}

Answer:"""


def build_prompt(query: str, context_block: str) -> str:
    """Fill the frozen template with a query and its labeled context block.

    `context_block` is the string returned by `rag.context.build_context` -- one
    labeled, truncated snippet per top-k document.
    """
    return PROMPT_TEMPLATE.format(query=query, context_block=context_block)
