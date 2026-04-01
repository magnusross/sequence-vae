import random

import torch
import torch.nn.functional as F

from .config import Config
from .data import decode_sequence, get_dataloaders
from .model import VAE


def vae_loss(logits, targets, mu, log_var, beta, pad_idx, free_bits=0.0):
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
    )
    # KL per dimension: [B, latent_dim] -> mean over batch -> [latent_dim]
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=0)
    # Free bits: each dimension must contribute at least free_bits nats
    kl = kl_per_dim.clamp(min=free_bits).sum()
    return recon + beta * kl, recon, kl_per_dim.sum().detach()


def word_dropout(x: torch.Tensor, rate: float, pad_idx: int) -> torch.Tensor:
    """Replace non-PAD tokens with pad_idx with probability `rate`."""
    if rate <= 0.0:
        return x
    mask = (torch.rand_like(x, dtype=torch.float) < rate) & (x != pad_idx)
    return x.masked_fill(mask, pad_idx)


def train_epoch(model, loader, optimizer, cfg, beta, device):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    for (x,) in loader:
        x = x.to(device)
        padding_mask = x == cfg.pad_idx
        dec_input = word_dropout(x, cfg.word_dropout_rate, cfg.pad_idx)
        logits, mu, log_var = model(x, padding_mask, decoder_input=dec_input)
        loss, recon, kl = vae_loss(logits, x, mu, log_var, beta, cfg.pad_idx, cfg.free_bits)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def val_epoch(model, loader, cfg, beta, device):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    for (x,) in loader:
        x = x.to(device)
        padding_mask = x == cfg.pad_idx
        logits, mu, log_var = model(x, padding_mask)
        loss, recon, kl = vae_loss(logits, x, mu, log_var, beta, cfg.pad_idx, cfg.free_bits)
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def print_samples(model, cfg, idx_to_char, n=10):
    samples = model.sample(n, cfg.max_seq_len, cfg.temperature)
    names = [decode_sequence(s, idx_to_char) for s in samples]
    print(f"  Prior samples: {names}")


def print_reconstructions(model, val_loader, cfg, idx_to_char, n=5, device="cpu"):
    model.eval()
    (x,) = next(iter(val_loader))
    x = x[:n].to(device)
    padding_mask = x == cfg.pad_idx
    mu = model.encode(x, padding_mask)
    recons = model.decoder.generate(mu, cfg.max_seq_len, cfg.temperature, cfg.eos_idx)
    for i in range(n):
        orig = decode_sequence(x[i], idx_to_char)
        rec = decode_sequence(recons[i], idx_to_char)
        print(f"  {orig!r:20s} -> {rec!r}")


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, char_to_idx, idx_to_char = get_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = VAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.num_epochs + 1):
        beta = min(1.0, epoch / max(cfg.kl_anneal_epochs, 1)) * cfg.kl_weight_max
        t_loss, t_recon, t_kl = train_epoch(model, train_loader, optimizer, cfg, beta, device)
        v_loss, v_recon, v_kl = val_epoch(model, val_loader, cfg, beta, device)

        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs} | beta={beta:.3f} | "
            f"train recon={t_recon:.4f} kl={t_kl:.4f} | "
            f"val recon={v_recon:.4f} kl={v_kl:.4f} "
            f"(free_bits floor={cfg.free_bits * cfg.latent_dim:.2f})"
        )

        if epoch % 10 == 0:
            print_samples(model, cfg, idx_to_char, n=10)
            print_reconstructions(model, val_loader, cfg, idx_to_char, n=5, device=device)

    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
        },
        "model.pt",
    )
    print("Saved model.pt")


if __name__ == "__main__":
    main()
