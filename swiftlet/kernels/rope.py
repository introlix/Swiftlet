import torch

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: int = 1
) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs / rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1)
    )
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(
        1, 2
    )
    return x_out