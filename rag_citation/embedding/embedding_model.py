from sentence_transformers import SentenceTransformer
from rag_citation.base_model import EmbedddingBaseModel


class EmbeddingModel(EmbedddingBaseModel):
    """
    Class for generating sentence embeddings using a shared SentenceTransformer model.

    This class inherits from EmbedddingBaseModel to utilize a shared instance of a
    SentenceTransformer model. It provides a method for generating embeddings for a
    given sentence.

    Args:
        embedding_model (str, optional): Size of the embedding model.
                                         Choose from "sm" (small), "md" (medium), or "lg" (large).
                                         Defaults to "sm".
    """

    def __init__(self, embedding_model="sm"):
        super().__init__(embedding_model)

    def embedding(self, sentence: str):
        """
        Generates embeddings for a given sentence.

        Args:
            sentence (str): The sentence to embed.

        Returns:
            torch.Tensor: A tensor representing the sentence embedding.
        """
        embeddings = self.model.encode([sentence], convert_to_tensor=True)
        return embeddings
