
import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    """Combine token embeddings and learned positional embeddings."""

    def __init__(
        self,
        vocab_size: int = 50_257,
        embedding_dim: int = 256,
        context_length: int = 128,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")

        self.context_length = context_length

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                "Expected token_ids with shape (batch_size, sequence_length)."
            )

        sequence_length = token_ids.size(1)

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors
