import torch
from swiftlet.models.gemma.config import get_gemma_config
from swiftlet.models.gemma.tokenizer import Tokenizer
from swiftlet.models.gemma3.model import Gemma3ForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM

test_model = Gemma3ForCausalLM.from_pretrained("swiftlet/models/gemma3")

gemma3_config = get_gemma_config(variant="1b", tokenizer=Tokenizer(model_path="swiftlet/models/gemma/tokenizer.model"))

model = Gemma3ForCausalLM(gemma3_config)

# TODO: Replace with the actual path to your model weights
model.from_pretrained(model_path="path/to/your/model/weights")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

tokenizer = gemma3_config.tokenizer

prompt = "Once upon a time"
output_ids = model.generate(
    prompts=prompt,
    device=device,
    output_len=50,    # how many tokens to generate
    temperature=0.8,
    top_p=0.9,
    top_k=50,
)

# Decode the generated token IDs into text
decoded_output = tokenizer.decode(output_ids[0])

print("▶️", decoded_output)