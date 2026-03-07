import warnings

warnings.filterwarnings("ignore")
import logging
from typing import List, Dict, Optional
from collections import defaultdict

from rag_citation.schema import RagCitationOutput

logger = logging.getLogger(__name__)


class Inference:
    """
    A class to perform inference for RAG citation.

    Supports two methods:
      - "non-llm" (default): SpaCy NER + SentenceTransformers cosine similarity
      - "llm": LLM-based citation via LiteLLM with structured output

    Non-LLM Args:
        spacy_model: The spaCy model size ("sm", "md", "lg").
        embedding_model: The embedding model size or a custom BaseEmbeddingModel instance.
        therhold_value: The threshold for cosine similarity scoring.

    LLM Args:
        model: LiteLLM model identifier (e.g., "gpt-4o", "azure/gpt-4o").
        api_key: Optional API key for the LLM provider.
        temperature: LLM temperature (default 0.0).
        max_tokens: Maximum tokens for LLM response.
        **litellm_kwargs: Additional LiteLLM parameters (e.g., api_base, api_version).
    """

    def __init__(
        self,
        method: str = "non-llm",
        # Non-LLM parameters
        spacy_model: str = "sm",
        embedding_model="sm",
        therhold_value: float = 0.88,
        # LLM parameters
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **litellm_kwargs,
    ) -> None:
        self.method = method

        if method == "non-llm":
            from rag_citation.focus_word import FocusWord
            from rag_citation.pair import GeneratePair
            from rag_citation.score import Score
            from rag_citation.base_model import EmbeddingModel

            self._focus_word = FocusWord(spacy_model)
            self._generate_pair = GeneratePair(spacy_model)
            self._score = Score()
            if isinstance(embedding_model, str):
                self._embedding_model = EmbeddingModel(embedding_model)
            else:
                self._embedding_model = embedding_model
            self._therhold_value = therhold_value

        elif method == "llm":
            if model is None:
                raise ValueError(
                    "The 'model' parameter is required when method='llm'. "
                    "Example: model='gpt-4o' or model='anthropic/claude-sonnet-4-20250514'"
                )
            from rag_citation.llm import LLMCitation

            self._llm_citation = LLMCitation(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **litellm_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'non-llm' or 'llm'."
            )

    def __call__(self, cite_item, messages: Optional[list] = None) -> RagCitationOutput:
        """
        Performs inference on the given cite item.

        Args:
            cite_item: A CiteItem with answer and context.
            messages: (LLM method only) Optional conversation messages list.
                     If provided, citation instructions are appended to this
                     conversation for richer context.

        Returns:
            RagCitationOutput with citation, missing_word, and hallucination.
        """
        if self.method == "non-llm":
            return self._run_non_llm(cite_item)
        elif self.method == "llm":
            return self._run_llm(cite_item, messages)

    # ------------------------------------------------------------------ #
    #  Non-LLM pipeline
    # ------------------------------------------------------------------ #

    def _run_non_llm(self, cite_item) -> RagCitationOutput:
        """Execute the existing non-LLM pipeline (SpaCy + SentenceTransformers)."""
        from tqdm import tqdm

        focus_words = self._focus_word.get_focus_word(cite_item.answer)
        pair = self._generate_pair.pair(focus_words, cite_item)

        citated, less_than_therhold_value, finded_label = self._cite(
            focus_words, pair
        )

        not_find = self._find_missing_labels(finded_label, focus_words)

        results, not_found, found_word = self._find_label(
            focus_words, not_find, less_than_therhold_value
        )

        missing_word = self._find_missing_words(
            focus_words.combine, found_word + finded_label
        )
        citated.update(results)
        hallucination = len(missing_word) > 0

        return RagCitationOutput(
            self._making_cite_pair(cite_item, citated), missing_word, hallucination
        )

    def _get_label_and_type(self, focus_words, word: str) -> tuple:
        for item in focus_words.combine:
            if item["words"].lower() == word.lower():
                return item["label"], item["type"]
        return None, None

    def _cite(self, focus_words, pair: List[Dict]):
        from tqdm import tqdm

        citation = {}
        finded_label = []
        less_than_threshold_value = {}

        for x in tqdm(pair):
            word_label = []
            answer_embedding = self._embedding_model.embedding(x["answer_sentences"])
            document_embedding = self._embedding_model.embedding(
                x["document_sentences"]
            )
            cosine_score_ = self._score.cosine_score(
                answer_embedding, document_embedding
            )

            if cosine_score_ >= self._therhold_value:
                for _word in x["word"]:
                    label_word = self._get_label_and_type(focus_words, _word)
                    if label_word[1] == "ENTITY":
                        if not any(item["word"] == _word for item in word_label):
                            word_label.append(
                                {"word": _word, "entity_name": label_word[0]}
                            )
                        finded_label.append(_word)

                citation[x["_id"]] = {
                    "answer_sentences": x["answer_sentences"],
                    "document_sentences": x["document_sentences"],
                    "word": list(set(x["word"])),
                    "label": word_label,
                    "source_id": x["source_id"],
                    "score": cosine_score_,
                }
            else:
                for _word in x["word"]:
                    label_word = self._get_label_and_type(focus_words, _word)
                    if label_word[1] == "ENTITY" and _word not in finded_label:
                        less_than_threshold_value[x["_id"]] = {
                            "answer_sentences": x["answer_sentences"],
                            "document_sentences": x["document_sentences"],
                            "word": list(set(x["word"])),
                            "label": list(set(word_label)),
                            "source_id": x["source_id"],
                            "score": cosine_score_,
                        }

        return citation, less_than_threshold_value, list(set(finded_label))

    def _find_missing_labels(self, finded_label: list, focus_words):
        non_common_elements = [
            item for item in focus_words.only_entity_word if item not in finded_label
        ]
        return non_common_elements

    def _find_label(self, focus_words, not_find, less_than_therhold_value: dict) -> Dict:
        results = {}
        not_found = []
        found_word = []
        for word in not_find:
            max_score = -1
            selected_json_id = None
            selected_json_data = None
            for key, data in less_than_therhold_value.items():
                if any(word.lower() in w.lower() for w in data["word"]):
                    if data["score"] > max_score:
                        max_score = data["score"]
                        selected_json_id = key
                        data["score"] = 100
                        selected_json_data = data
            if selected_json_id and selected_json_data:
                word_label = []
                for _word in selected_json_data["word"]:
                    label_word = self._get_label_and_type(focus_words, _word)
                    if label_word[1] == "ENTITY":
                        if not any(item["word"] == _word for item in word_label):
                            word_label.append(
                                {"word": _word, "entity_name": label_word[0]}
                            )
                selected_json_data["label"] = word_label
                results[selected_json_id] = selected_json_data
                found_word.append(word)
            else:
                not_found.append(word)
        return results, not_found, found_word

    def _find_missing_words(self, focus_words, final_output):
        mandatory_labels = ["DATE", "MONEY", "CARDINAL", "ORDINAL", "QUANTITY", "TIME"]
        entity_words = [
            focus_word["words"]
            for focus_word in focus_words
            if focus_word["type"] == "ENTITY"
            and focus_word["label"] in mandatory_labels
        ]
        return list(set(entity_words) - set(final_output))

    def _get_meta(self, cite_item, source_id: str):
        return cite_item.meta.get(source_id, None)

    def _making_cite_pair(self, cite_item, data):
        combined_results = {}
        for key, value in data.items():
            answer = value["answer_sentences"]
            document = value["document_sentences"]
            source_id = value["source_id"]
            score = value["score"]
            label = value["label"]

            if answer in combined_results:
                combined_results[answer]["cite_document"].append(
                    {
                        "document": document,
                        "source_id": source_id,
                        "score": score,
                        "entity": label,
                        "meta": self._get_meta(cite_item, source_id),
                    }
                )
            else:
                combined_results[answer] = {
                    "answer_sentences": answer,
                    "cite_document": [
                        {
                            "document": document,
                            "source_id": source_id,
                            "entity": label,
                            "score": score,
                            "meta": self._get_meta(cite_item, source_id),
                        }
                    ],
                }

        return [
            {
                "answer_sentences": content["answer_sentences"],
                "cite_document": content["cite_document"],
            }
            for content in combined_results.values()
        ]

    # ------------------------------------------------------------------ #
    #  LLM pipeline
    # ------------------------------------------------------------------ #

    def _run_llm(self, cite_item, messages=None) -> RagCitationOutput:
        """Execute the LLM-based citation pipeline."""
        if messages:
            citation_response = self._llm_citation.generate_from_messages(
                messages=messages,
                answer=cite_item.answer,
                context=cite_item.context,
            )
        else:
            citation_response = self._llm_citation.generate(
                answer=cite_item.answer,
                context=cite_item.context,
            )

        logger.info(f"LLM citations: {citation_response}")

        citation_output = self._format_llm_citations(citation_response, cite_item)

        return RagCitationOutput(
            citation=citation_output,
            missing_word=[],
            hallucination=False,
        )

    def _format_llm_citations(self, citation_response, cite_item) -> list:
        """
        Convert LLM CitationResponse into the same output format as the
        non-LLM method for consistency.
        """
        grouped = defaultdict(list)
        for item in citation_response.citations:
            grouped[item.sentence].append(item.source_document)

        output = []
        for sentence, source_ids in grouped.items():
            cite_documents = []
            for source_id in source_ids:
                doc_text = ""
                for ctx in cite_item.context:
                    if ctx.get("source_id") == source_id:
                        doc_text = ctx.get("document", "")
                        break
                cite_documents.append(
                    {
                        "document": doc_text,
                        "source_id": source_id,
                        "score": None,
                        "entity": [],
                        "meta": cite_item.meta.get(source_id, None),
                    }
                )
            output.append(
                {
                    "answer_sentences": sentence,
                    "cite_document": cite_documents,
                }
            )

        return output
