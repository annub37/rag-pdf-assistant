SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use the context below to answer as completely as possible. "
    "The context may come from MULTIPLE documents — make sure to cover ALL of them. "
    "If the context contains partial or related information, use it to give the best answer you can. "
    "Only say 'I don't have enough information' if the context has absolutely nothing relevant. "
    "Cite the document and page number(s) when possible."
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
    # Group chunks by document so the LLM sees each document's content together
    from collections import defaultdict
    docs = defaultdict(list)
    for chunk in chunks:
        docs[chunk.get("file_id", "unknown")].append(chunk)

    context_parts = []
    for doc_idx, (file_id, doc_chunks) in enumerate(docs.items(), 1):
        # Sort chunks within each document by page number
        doc_chunks.sort(key=lambda c: c.get("page_number", 0))
        doc_parts = []
        for chunk in doc_chunks:
            page = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            doc_parts.append(f"[Page {page}] {text}")
        context_parts.append(f"--- Document {doc_idx} ---\n" + "\n\n".join(doc_parts))

    context_block = "\n\n".join(context_parts)

    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
