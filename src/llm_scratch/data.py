
from pathlib import Path
from urllib.request import urlopen

import tiktoken
import torch
from torch.utils.data import Dataset


DATASET_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50_257


def download_dataset(data_dir: Path) -> Path:
    """Download the book's sample text if it is not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "the-verdict.txt"

    if file_path.exists() and file_path.stat().st_size > 0:
        return file_path

    # Download to a temporary file to avoid leaving a partial dataset.
    temporary_path = file_path.with_suffix(".tmp")

    try:
        with urlopen(DATASET_URL, timeout=30) as response:
            content = response.read()

        # Validate the text before saving it.
        text = content.decode("utf-8")
        if not text.strip():
            raise ValueError("The downloaded dataset is empty.")

        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(file_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return file_path


def load_text(file_path: Path) -> str:
    """Read the dataset as UTF-8 text."""
    text = file_path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"Dataset is empty: {file_path}")

    return text


def tokenize(text: str) -> list[int]:
    """Convert text to GPT-2 BPE token IDs."""
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    return tokenizer.encode(text)


class GPTDataset(Dataset):
    """Create input/target sequences using a sliding window."""

    def __init__(
        self,
        token_ids: list[int],
        max_length: int,
        stride: int,
    ) -> None:
        if max_length <= 0 or stride <= 0:
            raise ValueError("max_length and stride must be positive.")

        if len(token_ids) <= max_length:
            raise ValueError(
                "Not enough tokens to create an input-target pair."
            )

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride

    def __len__(self) -> int:
        return (
            len(self.token_ids) - self.max_length - 1
        ) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        end = start + self.max_length

        inputs = self.token_ids[start:end]
        targets = self.token_ids[start + 1:end + 1]

        return inputs, targets
