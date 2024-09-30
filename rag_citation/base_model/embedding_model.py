from sentence_transformers import SentenceTransformer
from rag_citation.base_model.base import BaseEmbeddingModel


class EmbeddingModel(BaseEmbeddingModel):
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

    def __init__(self, embedding_model="sm"):
        if embedding_model == "sm":
            self.model = SentenceTransformer(
                "avsolatorio/GIST-small-Embedding-v0", revision=None
            )

        elif embedding_model == "md":
            self.model = SentenceTransformer(
                "avsolatorio/GIST-Embedding-v0", revision=None
            )

        elif embedding_model == "lg":
            self.model = SentenceTransformer(
                "avsolatorio/GIST-large-Embedding-v0", revision=None
            )

        else:
            print("Warning::please choose `small`, `medium`, or `large`")
            print("Warning::choosing default model: small")
            self.model = SentenceTransformer(
                "avsolatorio/GIST-small-Embedding-v0", revision=None
            )

    def embedding(self, sentence: str) -> list:
        embeddings = self.model.encode([sentence], convert_to_tensor=True)
        return embeddings
