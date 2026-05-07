import torch.nn as nn


class DistanceMLP(nn.Module):
    """Small MLP distance regressor used by NoMaD-style models."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(embedding_dim // 16, 1),
        )

    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        return self.network(x)
