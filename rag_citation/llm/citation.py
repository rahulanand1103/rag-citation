import logging
from typing import Optional

from rag_citation.llm.prompt import build_citation_prompt, CITATION_USER_TEMPLATE
from rag_citation.llm.schema import CitationResponse

logger = logging.getLogger(__name__)


class LLMCitation:
    """
    LLM-based citation generator using LiteLLM.

    Args:
        model: Model identifier in LiteLLM format
               (e.g., "gpt-4o", "azure/gpt-4o", "anthropic/claude-sonnet-4-20250514").
        api_key: Optional API key. If not provided, LiteLLM reads from
                 environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
        temperature: LLM temperature for citation generation (default 0.0).
        max_tokens: Maximum tokens for the LLM response.
        **litellm_kwargs: Additional LiteLLM parameters (e.g., api_base, api_version).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **litellm_kwargs,
    ):
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "The 'litellm' package is required for LLM-based citations. "
                "Install it with: pip install rag-citation[llm]"
            )

        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.litellm_kwargs = litellm_kwargs
        self._litellm = litellm

    def _call_llm(self, messages: list) -> CitationResponse:
        """
        Call LiteLLM with structured output and return parsed CitationResponse.

        Args:
            messages: List of message dicts for litellm.completion().

        Returns:
            CitationResponse parsed from the LLM's structured output.
        """
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": CitationResponse,
            **self.litellm_kwargs,
        }

        if self.api_key:
            completion_kwargs["api_key"] = self.api_key

        response = self._litellm.completion(**completion_kwargs)
        raw_content = response.choices[0].message.content

        logger.info(f"LLM citation raw response: {raw_content}")

        return CitationResponse.model_validate_json(raw_content)

    def generate(self, answer: str, context: list) -> CitationResponse:
        """
        Generate citations by calling an LLM via LiteLLM.

        Args:
            answer: The LLM-generated answer text.
            context: List of dicts with 'source_id' and 'document' keys.

        Returns:
            CitationResponse with list of CitationItem.
        """
        messages = build_citation_prompt(answer, context)
        return self._call_llm(messages)

    def generate_from_messages(
        self, messages: list, answer: str, context: list
    ) -> CitationResponse:
        """
        Generate citations using a pre-existing conversation messages list.
        Appends citation instructions to the existing conversation.

        Useful when the user wants to pass the full conversation that produced
        the answer, giving the LLM more context for accurate citations.

        Args:
            messages: Existing conversation messages (list of role/content dicts).
            answer: The LLM-generated answer text.
            context: List of dicts with 'source_id' and 'document' keys.

        Returns:
            CitationResponse with list of CitationItem.
        """
        documents_text = ""
        for item in context:
            source_id = item.get("source_id", "unknown")
            document = item.get("document", "")
            documents_text += f"[source_id: {source_id}]\n{document}\n\n"

        citation_instruction = CITATION_USER_TEMPLATE.format(
            answer=answer,
            documents=documents_text.strip(),
        )

        augmented_messages = list(messages) + [
            {"role": "user", "content": citation_instruction}
        ]

        return self._call_llm(augmented_messages)
