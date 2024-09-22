from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embedding(self, sentence: str):
        """
        Generate embeddings for a given sentence.

        Args:
            sentence (str): The sentence to embed.

        Returns:
            Embeddings for the given sentence.
        """
        pass
