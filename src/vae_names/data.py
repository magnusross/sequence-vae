import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, dict, dict]:
    """Returns (train_loader, val_loader, char_to_idx, idx_to_char)"""
    names_path = Path("names.txt")
    if not names_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, names_path)

    with open(names_path) as f:
        names = [line.strip().lower() for line in f if line.strip()]

    char_to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for c in "abcdefghijklmnopqrstuvwxyz":
        char_to_idx[c] = len(char_to_idx)
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    def encode(name: str) -> list[int]:
        tokens = [cfg.sos_idx] + [char_to_idx[c] for c in name] + [cfg.eos_idx]
        # Pad to max_seq_len
        tokens = tokens[: cfg.max_seq_len]
        tokens += [cfg.pad_idx] * (cfg.max_seq_len - len(tokens))
        return tokens

    encoded = [encode(n) for n in names]
    data = torch.tensor(encoded, dtype=torch.long)

    n = len(data)
    n_train = int(0.9 * n)
    train_data = data[:n_train]
    val_data = data[n_train:]

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, char_to_idx, idx_to_char


def decode_sequence(ids: list[int] | torch.Tensor, idx_to_char: dict) -> str:
    """Strip special tokens and return the name string."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    chars = []
    for idx in ids:
        if idx in (0, 1):  # PAD or SOS
            continue
        if idx == 2:  # EOS
            break
        chars.append(idx_to_char.get(idx, "?"))
    return "".join(chars)
