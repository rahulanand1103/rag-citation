import spacy


class SpacyBaseModel:
    _nlp = None

    def __init__(self, spacy_model="sm"):
        """
        Base class for loading and sharing a SpaCy language model.

        This class ensures that only one instance of the specified SpaCy model is loaded,
        regardless of how many times the class is instantiated. This can save memory and time.

        Attributes:
            _nlp (spacy.Language): Static attribute to store the shared SpaCy language model.

        Args:
            spacy_model (str, optional): Size of the SpaCy model to load.
                                         Choose from "sm" (small), "md" (medium), or "lg" (large).
                                         Defaults to "sm".

        default model: `sm`
        """

        if SpacyBaseModel._nlp is None:

            if spacy_model == "sm":
                SpacyBaseModel._nlp = spacy.load("en_core_web_sm")

            elif spacy_model == "md":
                SpacyBaseModel._nlp = spacy.load("en_core_web_md")

            elif spacy_model == "lg":
                SpacyBaseModel._nlp = spacy.load("en_core_web_lg")

            else:
                print("Warning::Please provide correct input: sm, md, or lg")
                print("Warning::Running use sm(en_core_web_sm)")
                SpacyBaseModel._nlp = spacy.load("en_core_web_sm")

        self.nlp = SpacyBaseModel._nlp
