import re
import spacy
from typing import List, Dict
from rag_citation.base_model.spacy_model import SpacyBaseModel


class FindFocusWordInCiteData(SpacyBaseModel):
    """
    Class to find and highlight occurrences of focus words within text data.

    This class utilizes SpaCy to process text and identify sentences. It provides
    methods for finding occurrences of focus words in a list of documents and within
    a single answer string.

    Args:
        spacy_model (str, optional): Size of the SpaCy model to load.
                                     Choose from "sm" (small), "md" (medium), or "lg" (large).
                                     Defaults to "sm".
    """

    def __init__(self, spacy_model="sm") -> None:
        super().__init__(spacy_model)

    def find_focus_words_in_document(
        self, focus_words: List[Dict], documents: List[Dict]
    ) -> List[Dict]:
        """
        Finds occurrences of focus words within a list of documents.

        Args:
            focus_words (List[Dict]): A list of dictionaries, each containing "words", "label", and "type"
                                    keys representing the focus word data.
            documents (List[Dict]): A list of dictionaries, each containing "source_id" and "document" keys
                                   representing a document.

        Returns:
            List[Dict]: A list of dictionaries, each containing information about the focus word and its
                        occurrences in the documents.
        """
        results = []
        for document_data in documents:
            source_id = document_data["source_id"]
            document = document_data["document"]

            doc = self.nlp(document)
            sentences = [sent.text for sent in doc.sents]

            for focus_word_data in focus_words:
                word = focus_word_data["words"]
                label = focus_word_data["label"]
                word_type = focus_word_data["type"]

                word_occurrences = []

                for sentence in sentences:
                    start = 0
                    while True:
                        start = sentence.find(word, start)
                        if start == -1:
                            break
                        end = start + len(word)
                        word_occurrences.append(
                            {
                                "sentence": sentence,
                                "word_range": {"starting": start, "ending": end},
                                "source_id": source_id,
                            }
                        )
                        start += len(word)

                if word_occurrences:
                    results.append(
                        {
                            "type": word_type,
                            "label": label,
                            "word": word,
                            "sentences": word_occurrences,
                        }
                    )

        return results

    def find_focus_words_in_answer(
        self, focus_words: List[Dict], document: str
    ) -> List[Dict]:
        """
        Finds occurrences of focus words within a single answer string.

        Args:
            focus_words (List[Dict]): A list of dictionaries, each containing "words", "label", and "type"
                                        keys representing the focus word data.
            document (str): The answer string to search.

        Returns:
            List[Dict]: A list of dictionaries, each containing information about the focus word and its
                        occurrences in the answer.
        """
        results = []
        doc = self.nlp(document)
        sentences = [sent.text for sent in doc.sents]

        for focus_word_data in focus_words:
            word = focus_word_data["words"]
            label = focus_word_data["label"]
            word_type = focus_word_data["type"]

            word_occurrences = []

            for sentence in sentences:
                start = 0
                while True:
                    start = sentence.find(word, start)
                    if start == -1:
                        break
                    end = start + len(word)
                    word_occurrences.append(
                        {
                            "sentence": sentence,
                            "word_range": {"starting": start, "ending": end},
                        }
                    )
                    start += len(word)

            if word_occurrences:
                results.append(
                    {
                        "type": word_type,
                        "label": label,
                        "word": word,
                        "sentences": word_occurrences,
                    }
                )

        return results
