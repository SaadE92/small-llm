
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm_scratch.data import (
    GPTDataset,
    VOCAB_SIZE,
    download_dataset,
    load_text,
    tokenize,
)
from llm_scratch.embeddings import TokenAndPositionEmbedding


def main() -> None:
    torch.manual_seed(123)

    project_dir = Path(__file__).resolve().parent
    data_path = download_dataset(project_dir / "data")

    # 1. Load and tokenize the complete text.
    text = load_text(data_path)
    token_ids = tokenize(text)

    print(f"Characters: {len(text):,}")
    print(f"Tokens: {len(token_ids):,}")
    print(f"First 20 token IDs: {token_ids[:20]}")

    # 2. Split by token position, keeping the order of the text.
    split_index = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_index]
    val_ids = token_ids[split_index:]

    # 3. Build fixed-length input/target examples.
    context_length = 32
    stride = 32
    batch_size = 8

    train_dataset = GPTDataset(
        train_ids,
        max_length=context_length,
        stride=stride,
    )
    val_dataset = GPTDataset(
        val_ids,
        max_length=context_length,
        stride=stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # 4. Inspect a batch.
    inputs, targets = next(iter(train_loader))

    print(f"\nInput batch shape: {tuple(inputs.shape)}")
    print(f"Target batch shape: {tuple(targets.shape)}")
    print(f"First input:  {inputs[0].tolist()}")
    print(f"First target: {targets[0].tolist()}")

    # 5. Create and test token + position embeddings.
    embedding_model = TokenAndPositionEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=256,
        context_length=context_length,
    )

    embeddings = embedding_model(inputs)

    print(f"\nEmbedding output shape: {tuple(embeddings.shape)}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Embedding device: {embeddings.device}")
    print(f"Validation batches: {len(val_loader)}")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in embedding_model.parameters()
            if p.requires_grad),
    )

    assert embeddings.shape == (batch_size, context_length, 256)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()

    print("\nPipeline OK.")


if __name__ == "__main__":
    main()
