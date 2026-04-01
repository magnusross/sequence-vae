import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def sinusoidal_pos_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Returns [seq_len, d_model] sinusoidal positional encodings."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [S, D]


class Encoder(nn.Module):
    """
    Embeds input tokens, adds sinusoidal pos encoding, runs through TransformerEncoder,
    mean-pools non-padded positions, then projects to mu and log_var.

    Forward(x, src_key_padding_mask) -> (mu, log_var)
        x: [B, S] token ids
        src_key_padding_mask: [B, S] bool, True = padding position
        Returns: mu, log_var each [B, latent_dim]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim, padding_idx=cfg.pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)
        self.mu_head = nn.Linear(cfg.model_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.model_dim, cfg.latent_dim)

        # Register pos encoding as buffer (not a parameter)
        pe = sinusoidal_pos_encoding(cfg.max_seq_len, cfg.model_dim)
        self.register_buffer("pos_enc", pe)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor):
        # x: [B, S], mask: [B, S] True=padding
        B, S = x.shape
        emb = self.embed(x) + self.pos_enc[:S].unsqueeze(0)  # [B, S, D]
        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)  # [B, S, D]

        # Mean pool over non-padded positions
        not_pad = ~src_key_padding_mask  # [B, S] True=real token
        not_pad_f = not_pad.unsqueeze(-1).float()  # [B, S, 1]
        pooled = (out * not_pad_f).sum(dim=1) / not_pad_f.sum(dim=1).clamp(min=1)  # [B, D]

        mu = self.mu_head(pooled)
        log_var = self.logvar_head(pooled)
        return mu, log_var


class Decoder(nn.Module):
    """
    Autoregressive decoder conditioned on latent z.

    Projects z to a single token and prepends it to the embedded sequence.
    Uses causal self-attention (decoder-only transformer).

    Forward(z, tgt_seq) -> logits [B, S, vocab_size]
        Alignment:
          Transformer input:  [z_tok, emb(SOS), emb(c1), ..., emb(c_{n-1})]  length S+1
          Loss targets:       [SOS,   c1,        c2,   ...,   c_n          ]  length S

    Generate(z, max_len, temperature) -> [B, max_len] token ids
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.model_dim)   # z -> prepended token
        self.z_cond = nn.Linear(cfg.latent_dim, cfg.model_dim)   # z -> broadcast bias on all positions
        self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim, padding_idx=cfg.pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_decoder_layers)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.vocab_size)

        pe = sinusoidal_pos_encoding(cfg.max_seq_len + 1, cfg.model_dim)
        self.register_buffer("pos_enc", pe)

    def forward(self, z: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim]
        tgt_seq: [B, S] token ids (teacher-forced input, starts with SOS)
        Returns: logits [B, S, vocab_size]
        """
        B, S = tgt_seq.shape
        z_tok = self.z_proj(z).unsqueeze(1)  # [B, 1, D]
        tgt_emb = self.embed(tgt_seq)         # [B, S, D]

        # Prepend z token; add positional encoding
        seq = torch.cat([z_tok, tgt_emb], dim=1)  # [B, S+1, D]
        seq = seq + self.pos_enc[: S + 1].unsqueeze(0)
        # Broadcast z as a bias to every position so the decoder can't ignore it
        seq = seq + self.z_cond(z).unsqueeze(1)  # [B, 1, D] broadcast over S+1

        # Causal mask of size (S+1) x (S+1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S + 1, device=z.device)

        out = self.transformer(seq, mask=causal_mask, is_causal=True)  # [B, S+1, D]

        # Positions 0..S-1 of the output predict tokens at positions 0..S-1 of the target
        logits = self.out_proj(out[:, :S, :])  # [B, S, vocab_size]
        return logits

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_len: int,
        temperature: float = 1.0,
        eos_idx: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive generation starting from z token only.
        z: [B, latent_dim]
        Returns: [B, max_len] token ids (padded with eos_idx after first EOS)
        """
        B = z.shape[0]
        device = z.device
        z_tok = self.z_proj(z).unsqueeze(1)    # [B, 1, D]
        z_bias = self.z_cond(z).unsqueeze(1)  # [B, 1, D] broadcast bias

        # seq holds generated embeddings; token_ids holds sampled indices
        seq = z_tok + self.pos_enc[0].unsqueeze(0).unsqueeze(0) + z_bias  # [B, 1, D]
        token_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            S = seq.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            out = self.transformer(seq, mask=causal_mask, is_causal=True)  # [B, S, D]
            logits = self.out_proj(out[:, -1, :])  # [B, vocab_size]

            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

            # Once done, keep emitting eos_idx
            next_tok = torch.where(done, torch.full_like(next_tok, eos_idx), next_tok)
            token_ids[:, t] = next_tok
            done = done | (next_tok == eos_idx)

            # Embed next token and append
            next_emb = self.embed(next_tok).unsqueeze(1)  # [B, 1, D]
            next_emb = next_emb + self.pos_enc[S].unsqueeze(0).unsqueeze(0) + z_bias
            seq = torch.cat([seq, next_emb], dim=1)

            if done.all():
                break

        return token_ids


class VAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def _padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, S] bool mask: True where x == pad_idx."""
        return x == self.cfg.pad_idx

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        decoder_input: torch.Tensor | None = None,
    ):
        """
        x: [B, S] token ids (starts with SOS) — used for encoding and as loss target.
        decoder_input: optionally corrupted version of x for decoder input only.
                       If None, uses x unchanged.
        Returns: (logits [B, S, vocab_size], mu [B, L], log_var [B, L])
        """
        if padding_mask is None:
            padding_mask = self._padding_mask(x)
        mu, log_var = self.encoder(x, padding_mask)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(z, x if decoder_input is None else decoder_input)
        return logits, mu, log_var

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Deterministic encode: returns mu only."""
        if padding_mask is None:
            padding_mask = self._padding_mask(x)
        mu, _ = self.encoder(x, padding_mask)
        return mu

    @torch.no_grad()
    def sample(self, n: int, max_len: int, temperature: float = 1.0) -> torch.Tensor:
        """Sample z ~ N(0,I), decode. Returns [n, max_len] token ids."""
        device = next(self.parameters()).device
        z = torch.randn(n, self.cfg.latent_dim, device=device)
        return self.decoder.generate(z, max_len, temperature, eos_idx=self.cfg.eos_idx)
