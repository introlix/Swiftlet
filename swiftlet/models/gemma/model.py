import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig


def get_gemma_config(architecture):
    return AutoConfig.from_pretrained(f"google/{architecture}")

if __name__ == "__main__":
    print("Available Gemma architectures:")
    model = "gemma-2b"
    print(get_gemma_config(model))