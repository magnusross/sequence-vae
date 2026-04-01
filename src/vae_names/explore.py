import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from .config import Config
from .data import decode_sequence, get_dataloaders
from .model import VAE


def _model_device(model: VAE) -> torch.device:
    return next(model.parameters()).device


def load_model(path="model.pt") -> tuple[VAE, Config, dict, dict]:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    model = VAE(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg, char_to_idx, idx_to_char


def _encode_names(
    model: VAE, names: list[str], char_to_idx: dict, cfg: Config
) -> torch.Tensor:
    """Encode a list of name strings to z-space (mu). Returns [N, latent_dim]."""
    tokens = []
    for name in names:
        t = (
            [cfg.sos_idx]
            + [char_to_idx[c] for c in name.lower() if c in char_to_idx]
            + [cfg.eos_idx]
        )
        t = t[: cfg.max_seq_len]
        t += [cfg.pad_idx] * (cfg.max_seq_len - len(t))
        tokens.append(t)
    x = torch.tensor(tokens, dtype=torch.long, device=_model_device(model))
    return model.encode(x)


def sample_from_prior(
    model: VAE, cfg: Config, idx_to_char: dict, n: int = 20, temperature: float = 0.8
):
    """Sample z ~ N(0,I), decode. Print names."""
    seqs = model.sample(n, cfg.max_seq_len, temperature)
    names = [decode_sequence(s, idx_to_char) for s in seqs]
    print("Prior samples:")
    for name in names:
        print(f"  {name}")
    return names


def reconstruct(
    model: VAE, names: list[str], char_to_idx: dict, idx_to_char: dict, cfg: Config
):
    """Encode -> decode using mu (no noise). Print original vs reconstructed."""
    mu = _encode_names(model, names, char_to_idx, cfg)
    seqs = model.decoder.generate(
        mu, cfg.max_seq_len, temperature=0.5, eos_idx=cfg.eos_idx
    )
    print("Reconstructions:")
    for orig, seq in zip(names, seqs):
        rec = decode_sequence(seq, idx_to_char)
        print(f"  {orig!r:20s} -> {rec!r}")


def interpolate(
    model: VAE,
    name1: str,
    name2: str,
    char_to_idx: dict,
    idx_to_char: dict,
    cfg: Config,
    steps: int = 10,
    temperature: float = 0.5,
):
    """Lerp between z1 and z2. Print names at each step."""
    z = _encode_names(model, [name1, name2], char_to_idx, cfg)
    z1, z2 = z[0], z[1]
    alphas = torch.linspace(0, 1, steps, device=z.device)
    zs = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])  # [steps, latent_dim]
    seqs = model.decoder.generate(zs, cfg.max_seq_len, temperature, cfg.eos_idx)
    print(f"Interpolation {name1!r} -> {name2!r}:")
    for i, seq in enumerate(seqs):
        name = decode_sequence(seq, idx_to_char)
        print(f"  [{i:2d}] {name}")


def plot_latent_space(model: VAE, dataloader, idx_to_char: dict, cfg: Config):
    """Scatter of all encoded mus colored by first letter and name length."""
    all_mu = []
    all_first = []
    all_lengths = []

    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(_model_device(model))
            padding_mask = x == cfg.pad_idx
            mu = model.encode(x, padding_mask)
            all_mu.append(mu)
            for seq in x:
                name = decode_sequence(seq, idx_to_char)
                all_first.append(name[0] if name else "?")
                all_lengths.append(len(name))

    mu_np = torch.cat(all_mu, dim=0).cpu().numpy()
    first_letters = all_first
    lengths = np.array(all_lengths)

    # --- Plot by first letter ---
    unique_letters = sorted(set(first_letters))
    cmap = cm.get_cmap("tab20", len(unique_letters))
    letter_to_idx = {l: i for i, l in enumerate(unique_letters)}

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [letter_to_idx[l] for l in first_letters]
    sc = ax.scatter(mu_np[:, 0], mu_np[:, 1], c=colors, cmap="tab20", s=5, alpha=0.6)
    # Legend with letter labels
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(letter_to_idx[l]),
            markersize=6,
            label=l,
        )
        for l in unique_letters
    ]
    ax.legend(handles=handles, title="First letter", loc="best", fontsize=7, ncol=3)
    ax.set_title("Latent space by first letter")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    fig.savefig("latent_by_letter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved latent_by_letter.png")

    # --- Plot by name length ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(mu_np[:, 0], mu_np[:, 1], c=lengths, cmap="viridis", s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Name length")
    ax.set_title("Latent space by name length")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    fig.savefig("latent_by_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved latent_by_length.png")


def decode_grid(
    model: VAE,
    cfg: Config,
    idx_to_char: dict,
    grid_size: int = 15,
    z_range: tuple[float, float] = (-3, 3),
    temperature: float = 0.5,
):
    """Decode a grid of z points and print as 2D table."""
    vals = torch.linspace(
        z_range[0], z_range[1], grid_size, device=_model_device(model)
    )
    # Build all (z0, z1) combinations; z0 = x-axis (columns), z1 = y-axis (rows, top=high)
    z0, z1 = torch.meshgrid(vals, vals, indexing="xy")
    zs = torch.stack([z0.flatten(), z1.flatten()], dim=1)  # [grid_size^2, 2]

    with torch.no_grad():
        seqs = model.decoder.generate(zs, cfg.max_seq_len, temperature, cfg.eos_idx)

    names = [decode_sequence(s, idx_to_char) for s in seqs]
    grid = [names[i * grid_size : (i + 1) * grid_size] for i in range(grid_size)]

    print(f"Latent grid ({grid_size}x{grid_size}), z in {z_range}:")
    # Print rows from top (high z1) to bottom (low z1)
    for row in reversed(grid):
        print("  " + "  ".join(f"{n:12s}" for n in row))


def explore_neighborhood(
    model: VAE,
    name: str,
    char_to_idx: dict,
    idx_to_char: dict,
    cfg: Config,
    noise_scale: float = 0.3,
    n: int = 15,
    temperature: float = 0.5,
):
    """Perturb z_mu with small noise and decode variants."""
    mu = _encode_names(model, [name], char_to_idx, cfg)  # [1, L]
    noise = torch.randn(n, cfg.latent_dim, device=mu.device) * noise_scale
    zs = mu.expand(n, -1) + noise  # [n, L]
    with torch.no_grad():
        seqs = model.decoder.generate(zs, cfg.max_seq_len, temperature, cfg.eos_idx)
    variants = [decode_sequence(s, idx_to_char) for s in seqs]
    print(f"Neighborhood of {name!r} (noise_scale={noise_scale}):")
    for v in variants:
        print(f"  {v}")


def main():
    model, cfg, char_to_idx, idx_to_char = load_model("model.pt")
    _, val_loader, _, _ = get_dataloaders(cfg)

    print("=" * 60)
    sample_from_prior(model, cfg, idx_to_char, n=20, temperature=0.8)

    print("=" * 60)
    test_names = ["emma", "oliver", "sophia", "james", "ava"]
    reconstruct(model, test_names, char_to_idx, idx_to_char, cfg)

    print("=" * 60)
    interpolate(
        model, "alice", "bob", char_to_idx, idx_to_char, cfg, steps=10, temperature=0.5
    )

    print("=" * 60)
    decode_grid(model, cfg, idx_to_char, grid_size=10, z_range=(-3, 3), temperature=0.5)

    print("=" * 60)
    explore_neighborhood(
        model,
        "emma",
        char_to_idx,
        idx_to_char,
        cfg,
        noise_scale=0.3,
        n=15,
        temperature=0.5,
    )

    print("=" * 60)
    print("Plotting latent space (this may take a moment)...")
    plot_latent_space(model, val_loader, idx_to_char, cfg)


if __name__ == "__main__":
    main()
