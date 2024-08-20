from sentence_transformers import SentenceTransformer


class EmbedddingBaseModel:
    """
    Base class for embedding models.

    This class provides a shared instance of a SentenceTransformer model based on the specified size.

    Attributes:
        _model (SentenceTransformer): Static attribute to store the shared SentenceTransformer instance.

    Args:
        embedding_model (str, optional): Size of the embedding model.
                                         Choose from "sm" (small), "md" (medium), or "lg" (large).
                                         Defaults to "sm".

    default model: `sm`
    """

    _model = None

    def __init__(self, embedding_model="sm"):

        if EmbedddingBaseModel._model is None:

            if embedding_model == "sm":
                EmbedddingBaseModel._model = SentenceTransformer(
                    "avsolatorio/GIST-small-Embedding-v0", revision=None
                )

            elif embedding_model == "md":
                EmbedddingBaseModel._model = SentenceTransformer(
                    "avsolatorio/GIST-Embedding-v0", revision=None
                )

            elif embedding_model == "lg":
                EmbedddingBaseModel._model = SentenceTransformer(
                    "avsolatorio/GIST-large-Embedding-v0", revision=None
                )

            else:
                print("Warning::please choose `small`, `medium`, or `large`")
                print("Warning::choosing default model: small")
                EmbedddingBaseModel._model = SentenceTransformer(
                    "avsolatorio/GIST-small-Embedding-v0", revision=None
                )

        self.model = EmbedddingBaseModel._model
