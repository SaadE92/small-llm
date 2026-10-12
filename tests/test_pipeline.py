
import torch
import pytest

from llm_scratch.data import GPTDataset, tokenize
from llm_scratch.embeddings import TokenAndPositionEmbedding


def test_tokenization_returns_integer_ids():
    ids = tokenize("Hello world")

    assert len(ids) > 0
    assert all(isinstance(token_id, int) for token_id in ids)


def test_dataset_shifts_targets_by_one_token():
    dataset = GPTDataset(
        token_ids=[10, 20, 30, 40, 50, 60],
        max_length=4,
        stride=1,
    )

    inputs, targets = dataset[0]

    assert inputs.tolist() == [10, 20, 30, 40]
    assert targets.tolist() == [20, 30, 40, 50]


def test_embedding_output_shape():
    model = TokenAndPositionEmbedding(
        vocab_size=100,
        embedding_dim=16,
        context_length=8,
    )

    token_ids = torch.randint(0, 100, (2, 5))
    output = model(token_ids)

    assert output.shape == (2, 5, 16)
    assert torch.isfinite(output).all()


def test_embedding_rejects_long_sequences():
    model = TokenAndPositionEmbedding(
        vocab_size=100,
        embedding_dim=16,
        context_length=8,
    )

    token_ids = torch.zeros((2, 9), dtype=torch.long)

    with pytest.raises(ValueError):
        model(token_ids)
