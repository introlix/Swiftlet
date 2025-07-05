import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        # if quant:
        #     self.weight = nn.Parameter(
        #         torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
        #         requires_grad=False,
        #     )
        #     self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        # else:
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim)),
            requires_grad=False,
        )
        # self.quant = quant

    def forward(self, x):
        weight = self.weight
        # if self.quant:
        #     weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output