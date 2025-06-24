from swiftlet.models.gemma import config
from swiftlet.models.gemma.tokenizer import Tokenizer

gemma3_config = config.get_gemma_config(variant="1b", tokenizer=Tokenizer(model_path="swiftlet/models/gemma/tokenizer.model"))