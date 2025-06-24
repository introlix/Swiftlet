import torch
from swiftlet.models.gemma import config
from swiftlet.models.gemma.tokenizer import Tokenizer
from swiftlet.models.gemma3.model import Gemma3ForCausalLM

gemma3_config = config.get_gemma_config(variant="1b", tokenizer=Tokenizer(model_path="swiftlet/models/gemma/tokenizer.model"))

model = Gemma3ForCausalLM(config)

model.from_pretrained(model_path="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

prompt = "Once upon a time"
output = model.generate(
    prompts=prompt,
    device=device,
    output_len=50,    # how many tokens to generate
    temperature=0.8,
    top_p=0.9,
    top_k=50,
)


print("▶️", output)