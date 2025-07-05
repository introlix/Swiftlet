import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant: bool, quant_type: str, bias: bool):
        super().__init__()
        self.quant = quant
        if not quant:
            if quant_type:
                raise ValueError("quant_type should not be specified when quant is False.")
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        else:
            if not quant_type:
                raise ValueError("quant_type must be specified when quant is True. Available options: 'int8', 'int4' 'fp4'.")
            if quant_type == 'int8':
                self.linear = bnb.nn.Linear8bitLt(
                    in_features,
                    out_features,
                    bias=bias,
                    has_fp16_weights=True
                )
            elif quant_type in ("int4", "nf4", "fp4"):
                qtype = "nf4" if quant_type == "int4" else quant_type
                self.linear = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    bias=bias,
                    quant_type=qtype
                )
            else:
                raise ValueError(
                    "quant_type must be 'int4', 'int8', 'fp4' when quant=True"
                )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights for standard linear; bnb modules handle their own init."""
        if not self.quant:
            self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)

