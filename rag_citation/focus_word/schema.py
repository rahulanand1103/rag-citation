from dataclasses import dataclass
from typing import List, Dict


@dataclass
class FocusWordDataType:
    """
    Dataclass to store categorized focus words extracted from text.

    Attributes:
        only_entity_word: A list of dictionaries, each representing a named entity
                          with its text and label.
        only_normal_word: A list of dictionaries, each representing a noun with its text and POS tag.
        combine: A combined list of entity and noun dictionaries.
    """

    only_entity_word: List[Dict]
    only_normal_word: List[Dict]
    combine: List[Dict]
