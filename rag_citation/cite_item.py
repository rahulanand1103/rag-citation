import re


class CiteItem:
    """
    Data structure to hold an answer, its corresponding context, and metadata.

    Args:
        answer (str): The answer extracted from the context.
        context (list): A list of dictionaries, each representing a source of information.
                        Each dictionary must contain:
                            - 'document': str, The source text.
                        Each dictionary may optionally contain:
                            - 'source_id': str, A unique identifier for the source.
                            - 'meta': dict, Additional metadata associated with the source.

    Attributes:
        answer (str): The answer extracted from the context.
        context (list): The list of context dictionaries.
        meta (dict): A dictionary mapping source_ids to their corresponding metadata.

    Raises:
        ValueError: If answer is not a non-empty string, context is not a non-empty list,
                    or any item in the context list doesn't contain a 'document' key.

    Example:
        >>> context = [
        ...     {"document": "This is the first sentence.", "source_id": "doc1", "meta": {"doc_id": "doc_id"}},
        ...     {"document": "This is the second sentence.", "source_id": "doc2", "meta": {"doc_id": "doc_id"}},
        ... ]
        >>> cite_item = CiteItem(answer="some answer", context=context)
        >>> print(cite_item.meta)
    """

    def __init__(self, answer: str, context: list):
        # Validate answer input
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")

        # Validate context input
        if not isinstance(context, list) or not context:
            raise ValueError("Context must be a non-empty list.")

        required_keys = {"document"}
        for item in context:
            if not isinstance(item, dict):
                raise ValueError("Each context item must be a dictionary.")
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                raise ValueError(
                    f"Each context item must contain the following keys: {', '.join(missing_keys)}"
                )

        # Assign attributes
        self.answer = self._clean_text(answer)
        self.context = [self._clean_text(item) for item in context]
        self.meta = {item.get("source_id"): item.get("meta") for item in context}

    def _clean_text(self, text):
        # Clean the input text by removing newline characters
        if isinstance(text, str):
            # text = text.replace("\n", "\n ")
            text = re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)
            return text
        elif isinstance(text, dict):
            return {k: self._clean_text(v) for k, v in text.items()}
        else:
            return text

    def __str__(self):
        # String representation for easy inspection
        attributes = ", ".join(f"{key}={getattr(self, key)}" for key in self.__dict__)
        return f"CiteItem({attributes})"
