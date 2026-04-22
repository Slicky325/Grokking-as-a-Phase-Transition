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


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=1,
                 dim_feedforward=512, max_len=5, init_scale=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.unembed = nn.Linear(d_model, vocab_size)

        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(init_scale)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded)
        return self.unembed(output[:, -1, :])


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
