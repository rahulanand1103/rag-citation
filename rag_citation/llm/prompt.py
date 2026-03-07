CITATION_SYSTEM_PROMPT = """You are a citation assistant. Your job is to analyze an AI-generated answer and identify which source documents support each sentence in the answer.

You will be given:
1. An answer (composed of one or more sentences)
2. A list of source documents, each with a source_id

For each sentence in the answer, determine which source document(s) it is derived from or supported by. Only cite a sentence if you can find clear supporting evidence in a source document. If a sentence is not supported by any source document, do not include it in the citations."""

CITATION_USER_TEMPLATE = """Answer:
{answer}

Source Documents:
{documents}

Analyze the answer and identify which source document supports each sentence. Return the citations as structured output."""


def build_citation_prompt(answer: str, context: list) -> list:
    """
    Build the messages list for the LLM citation call.

    Args:
        answer: The LLM-generated answer text.
        context: List of dicts, each with 'source_id' and 'document' keys.

    Returns:
        A list of message dicts suitable for litellm.completion().
    """
    documents_text = ""
    for item in context:
        source_id = item.get("source_id", "unknown")
        document = item.get("document", "")
        documents_text += f"[source_id: {source_id}]\n{document}\n\n"

    user_content = CITATION_USER_TEMPLATE.format(
        answer=answer,
        documents=documents_text.strip(),
    )

    return [
        {"role": "system", "content": CITATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
