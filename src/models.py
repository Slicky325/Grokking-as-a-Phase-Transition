import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        return x + self.pos_emb(positions)


class _NoNormTransformerLayer(nn.Module):
    """Single transformer block without LayerNorm — matching the reference
    progress-measures-paper architecture (use_ln=False).  Uses pre-residual
    layout: attn and MLP outputs are added directly to the residual stream."""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff1  = nn.Linear(d_model, dim_feedforward)
        self.ff2  = nn.Linear(dim_feedforward, d_model)
        self.act  = nn.ReLU()

    def forward(self, x):
        # is_causal=True must NOT be combined with an explicit attn_mask (PyTorch 2.x);
        # use the explicit lower-triangular causal mask only.
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff2(self.act(self.ff1(x)))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=1,
                 dim_feedforward=512, max_len=5, init_scale=1.0, use_ln=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model

        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        if use_ln:
            # Standard PyTorch pre-norm transformer (original behaviour)
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                batch_first=True, norm_first=True,
            )
            self.blocks = nn.ModuleList(
                [nn.TransformerEncoder(encoder_layers, num_layers=1)]
            )
            self._use_ln = True
        else:
            # No LayerNorm — matches reference (Nanda et al. / devinterp)
            self.blocks = nn.ModuleList(
                [_NoNormTransformerLayer(d_model, nhead, dim_feedforward)
                 for _ in range(num_layers)]
            )
            self._use_ln = False

        self.unembed = nn.Linear(d_model, vocab_size)

        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(init_scale)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x[:, -1, :])


class SmallMLP(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden_size=512, init_scale=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)

        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(init_scale)

    def forward(self, x):
        embedded = self.embedding(x).sum(dim=1)
        out = self.act(self.fc1(embedded))
        return self.fc2(out)
