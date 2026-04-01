import dataclasses


@dataclasses.dataclass
class Config:
    # Vocab
    vocab_size: int = 29          # a-z + PAD(0) + SOS(1) + EOS(2)
    pad_idx: int = 0
    sos_idx: int = 1
    eos_idx: int = 2
    max_seq_len: int = 20

    # Model
    model_dim: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 1
    latent_dim: int = 2           # 2D for visualization
    dropout: float = 0.1
    ff_dim: int = 256

    # Training
    batch_size: int = 256
    lr: float = 3e-4
    num_epochs: int = 100
    kl_anneal_epochs: int = 40
    kl_weight_max: float = 1.0
    temperature: float = 0.8
    grad_clip: float = 1.0
    seed: int = 42
    word_dropout_rate: float = 0.8
    free_bits: float = 1.0        # min KL per latent dimension (nats)
