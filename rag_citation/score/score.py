import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch


class Score:
    """
    Calculates the cosine similarity score between two text embeddings.
    """

    def cosine_score(self, embeddings1, embeddings2):
        """
        Calculates the cosine similarity score between two text embeddings.

        Args:
            embeddings1 (torch.Tensor): The first embedding tensor.
            embeddings2 (torch.Tensor): The second embedding tensor.

        Returns:
            float: The cosine similarity score between the two embeddings.
        """
        scores = F.cosine_similarity(embeddings1, embeddings2, dim=-1)
        return scores.tolist()[0]
