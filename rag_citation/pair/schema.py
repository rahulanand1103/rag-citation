class FocusWordDataType:
    """
    Data structure to hold different representations of a focus word.

    Attributes:
        only_entity_word (str): The focus word with only named entities preserved.
        only_normal_word (str): The focus word with only non-named entities preserved.
        combine (str): The focus word with both named entities and non-named entities.
    """

    def __init__(self):
        self.only_entity_word = None
        self.only_normal_word = None
        self.combine = None


class CiteItem:
    """
    Data structure to hold an answer and its corresponding context.

    Attributes:
        answer (str): The answer extracted from the context.
        context (str): The text containing the answer.
    """

    def __init__(self):
        self.answer = None
        self.context = None
