SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use ONLY the context below to answer. "
    "If the answer is not in the context, say: 'I don't have enough information to answer that.' "
    "Cite the page number when possible."
)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for the OpenAI chat API.

    Args:
        query: The user's question.
        chunks: List of dicts with 'text' and 'page_number' from vector search.

    Returns:
        A list of message dicts: [{"role": ..., "content": ...}, ...]
    """
    # Format each chunk with its page number so the LLM can cite sources
    context_parts = []
    for chunk in chunks:
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        context_parts.append(f"[Page {page}] {text}")

    context_block = "\n\n".join(context_parts)

    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
