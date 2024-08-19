import spacy
from rag_citation.focus_word.schema import FocusWordDataType
from rag_citation.base_model import SpacyBaseModel


class FocusWord(SpacyBaseModel):
    """
    Class for extracting focus words (entities and nouns) from a text answer using SpaCy.

    This class inherits from SpacyBaseModel to utilize a shared instance of a SpaCy
    language model. It identifies named entities and nouns in the provided text and
    returns them in a structured format.

    Args:
        spacy_model (str, optional): Size of the SpaCy model to load.
                                     Choose from "sm" (small), "md" (medium), or "lg" (large).
                                     Defaults to "sm".
    """

    def __init__(self, spacy_model="sm") -> None:
        super().__init__(spacy_model)

    def get_focus_word(self, answer: str) -> FocusWordDataType:
        """
        Extracts and categorizes focus words from a text answer.

        Args:
            answer (str): The text answer to process.

        Returns:
            FocusWordDataType: A namedtuple containing:
                - entities_dict: A dictionary mapping entity text to its label.
                - focus_words: A list of dictionaries, each representing a noun with its text and POS tag.
                - combined_list: A combined list of entity and focus word dictionaries.
        """
        doc = self.nlp(answer)

        entities_list = [
            {"type": "ENTITY", "words": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]

        entities_dict = {ent["words"]: ent["label"] for ent in entities_list}

        focus_words = [
            {"type": "WORD", "words": token.text, "label": token.pos_}
            for token in doc
            if token.pos_ == "NOUN" and token.text not in entities_dict
        ]

        combined_list = entities_list + focus_words

        return FocusWordDataType(entities_dict, focus_words, combined_list)
