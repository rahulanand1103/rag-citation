from rag_citation.pair.focus_word_in_cite_data import FindFocusWordInCiteData
from rag_citation.pair.schema import FocusWordDataType, CiteItem
from collections import defaultdict
import uuid
from typing import List, Dict


class GeneratePair(FindFocusWordInCiteData):
    """
    Class to generate pairs of matching words and sentences from answers and documents.

    This class extends FindFocusWordInCiteData to identify common focus words between
    an answer and a set of documents. It groups these words based on their shared
    sentences and source IDs, creating unique pairings for further processing.

    Args:
        type (str, optional): Size of the SpaCy model to load.
                             Choose from "sm" (small), "md" (medium), or "lg" (large).
                             Defaults to "sm".
    """

    def __init__(self, type="sm") -> None:
        super().__init__(type)

    def _find_common_words(
        self, answer: List[Dict], document: List[Dict]
    ) -> List[Dict]:
        """
        Finds common words between the answer and document data.

        Args:
            answer (List[Dict]): Processed focus word data from the answer.
            document (List[Dict]): Processed focus word data from the documents.

        Returns:
            List[Dict]: A list of dictionaries, each containing a common word and its associated sentences
                        from the answer and document.
        """
        answer_words = {entry["word"]: entry for entry in answer}
        document_words = {entry["word"]: entry for entry in document}

        common_words = [
            {
                "word": word,
                "type": answer_entry["type"],
                "label": answer_entry["label"],
                "answer_sentences": answer_sentence["sentence"],
                "document_sentences": document_sentence["sentence"],
                "source_id": document_sentence["source_id"],
            }
            for word, answer_entry in answer_words.items()
            if word in document_words
            for answer_sentence in answer_entry["sentences"]
            for document_sentence in document_words[word]["sentences"]
        ]

        return common_words

    def _generate_uniqueid(self, length=4):
        """
        Generates a unique ID of a specified length.

        Args:
            length (int, optional): The desired length of the unique ID. Defaults to 4.

        Returns:
            str: A unique ID string.
        """
        uuid_int = uuid.uuid4().int
        uuid_str = str(uuid_int)
        unique_id = uuid_str[:length]
        return unique_id

    def _combine_words(self, data: List[Dict]) -> List[Dict]:
        """
        Combines words that share the same answer sentence, document sentence, and source ID.

        Args:
            data (List[Dict]): A list of dictionaries containing common word data.

        Returns:
            List[Dict]: A list of dictionaries where words sharing the same context are grouped together.
        """

        word_groups = defaultdict(list)

        for item in data:
            key = (
                item["answer_sentences"],
                item["document_sentences"],
                item["source_id"],
            )
            word_groups[key].append(item["word"])

        combined_words = [
            {
                "_id": self._generate_uniqueid(),
                "word": words,
                "answer_sentences": key[0],
                "document_sentences": key[1],
                "source_id": key[2],
            }
            for key, words in word_groups.items()
        ]

        return combined_words

    def pair(self, focus_words: FocusWordDataType, cite_item: CiteItem) -> List[Dict]:
        """
        Pairs focus words from the answer with occurrences in the cited documents.

        Args:
            focus_words (FocusWordDataType): Extracted focus words from the answer.
            cite_item (CiteItem): Citation data containing the answer and relevant documents.

        Returns:
            List[Dict]: A list of paired word data, including unique IDs, words, and associated sentences.
        """

        focus_answer = self.find_focus_words_in_answer(
            focus_words.combine, cite_item.answer
        )
        focus_context = self.find_focus_words_in_document(
            focus_words.combine, cite_item.context
        )

        common_words = self._find_common_words(focus_answer, focus_context)

        combined_output = self._combine_words(common_words)

        return combined_output
