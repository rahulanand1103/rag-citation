from dataclasses import dataclass
from typing import List, Dict


@dataclass
class RagCitationOutput:
    """
    A dataclass to store the output of a RAG citation model.

    Attributes:
        citation: A list of dictionaries, each representing a citation.
        missing_word: A list of dictionaries, each representing a missing word.
        hallucination: A boolean indicating whether the model hallucinated any information.
    """

    citation: List[Dict]
    missing_word: List[Dict]
    hallucination: bool
