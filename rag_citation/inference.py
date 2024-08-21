import warnings

warnings.filterwarnings("ignore")

from rag_citation.focus_word import FocusWordDataType, FocusWord
from rag_citation.pair import GeneratePair
from typing import List, Dict
from tqdm import tqdm

from rag_citation.embedding import EmbeddingModel
from rag_citation.score import Score
from rag_citation.schema import RagCitationOutput


class Inference(FocusWord, GeneratePair):
    """
    A class to perform inference on a RAG citation model.

    Attributes:
        spacy_model: The spaCy model to use for tokenization.
        embedding_model: The embedding model to use for semantic similarity.
        therhold_value: The threshold value for the cosine similarity score.
    """

    def __init__(
        self, spacy_model="sm", embedding_model="sm", therhold_value=0.88
    ) -> None:
        """
        Initializes the Inference class.

        Args:
            spacy_model: The spaCy model to use for tokenization.
            embedding_model: The embedding model to use for semantic similarity.
            therhold_value: The threshold value for the cosine similarity score.
        """
        super().__init__(spacy_model)
        self.score = Score()
        self.embedding_model = EmbeddingModel(embedding_model)
        self.therhold_value = therhold_value

    def _get_label_and_type(self, focus_words: FocusWordDataType, word: str) -> tuple:
        """
        Gets the label and type of a focus word.

        Args:
            focus_words: The focus words.
            word: The word to get the label and type for.

        Returns:
            A tuple containing the label and type of the word.
        """
        for item in focus_words.combine:
            if item["words"].lower() == word.lower():
                return item["label"], item["type"]
        return None, None

    def _cite(self, focus_words: FocusWordDataType, pair: List[Dict]):
        """
        Cites the focus words from the given pair of documents.

        Args:
            focus_words: The focus words.
            pair: The pair of documents.

        Returns:
            A tuple containing the cited documents, the documents with scores less than the threshold, and the found labels.
        """
        citation = {}
        finded_label = []
        less_than_threshold_value = {}

        for x in tqdm(pair):
            word_label = []
            answer_embedding = self.embedding_model.embedding(x["answer_sentences"])
            document_embedding = self.embedding_model.embedding(x["document_sentences"])
            cosine_score_ = self.score.cosine_score(
                answer_embedding, document_embedding
            )

            if cosine_score_ >= self.therhold_value:
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

    def _find_missing_labels(self, finded_label: list, focus_words: FocusWordDataType):
        """
        Finds the labels that were not found in the cited documents.

        Args:
            finded_label: The labels that were found in the cited documents.
            focus_words: The focus words.

        Returns:
            The labels that were not found in the cited documents.
        """
        non_common_elements = [
            item for item in focus_words.only_entity_word if item not in finded_label
        ]
        return non_common_elements

    def _find_label(
        self, focus_words, not_find, less_than_therhold_value: dict
    ) -> Dict:
        """
        Finds the labels that were not found in the cited documents but are present in the documents with scores less than the threshold.

        Args:
            focus_words: The focus words.
            not_find: The labels that were not found in the cited documents.
            less_than_therhold_value: The documents with scores less than the threshold.

        Returns:
            A dictionary containing the found labels, the labels that were not found, and the found words.
        """
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
        """
        Finds the words that were not found in the cited documents.

        Args:
            focus_words: The focus words.
            final_output: The final output of the model.

        Returns:
            The words that were not found in the cited documents.
        """
        mandatory_labels = ["DATE", "MONEY", "CARDINAL", "ORDINAL", "QUANTITY", "TIME"]

        # Extract entity words with allowed labels
        entity_words = [
            focus_word["words"]
            for focus_word in focus_words
            if focus_word["type"] == "ENTITY"
            and focus_word["label"] in mandatory_labels
        ]

        missing_entity_words = list(set(entity_words) - set(final_output))
        return missing_entity_words

    def _get_meta(self, cite_item: FocusWordDataType, source_id: str):
        """
        Gets the meta data for the given source ID.

        Args:
            cite_item: The cite item.
            source_id: The source ID.

        Returns:
            The meta data for the given source ID.
        """
        return cite_item.meta.get(source_id, None)

    def making_cite_pair(self, cite_item: FocusWordDataType, data):
        """
        Makes a pair of cited documents for the given cite item.

        Args:
            cite_item: The cite item.
            data: The cited documents.

        Returns:
            A list of pairs of cited documents.
        """
        # Initialize an empty dictionary to store the combined results
        combined_results = {}

        # Iterate over the data to combine entries with the same answer sentences
        for key, value in data.items():
            answer = value["answer_sentences"]
            document = value["document_sentences"]
            source_id = value["source_id"]
            label = value["label"]

            if answer in combined_results:
                combined_results[answer]["cite_document"].append(
                    {
                        "document": document,
                        "source_id": source_id,
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
                            "meta": self._get_meta(cite_item, source_id),
                        }
                    ],
                }

        # Convert the combined results to the desired output format
        output = []
        for answer, content in combined_results.items():
            output.append(
                {
                    "answer_sentences": content["answer_sentences"],
                    "cite_document": content["cite_document"],
                }
            )
        return output

    def __call__(self, cite_item: FocusWordDataType) -> RagCitationOutput:
        """
        Performs inference on the given cite item.

        Args:
            cite_item: The cite item.

        Returns:
            A RagCitationOutput object containing the citation, missing words, and hallucination status.
        """
        focus_words = self.get_focus_word(cite_item.answer)

        pair = self.pair(focus_words, cite_item)

        citated, less_than_therhold_value, finded_label = self._cite(focus_words, pair)

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
            self.making_cite_pair(cite_item, citated), missing_word, hallucination
        )
