from transformers import(
    AutoConfig,
    GPT2LMHeadModel
)
from configs.causal_config import ConfigModel

class CustomModel():
  def __init__(self,
               model_name=ConfigModel.MODEL_NAME,
               context_length=ConfigModel.CONTEXT_LENGTH,
               vocab_size=None,
               bos_token_id=None,
               eos_token_id=None,
               flag_training=True
               ) -> None:
    self.model_name = model_name
    self.model = self.create_model(self.setup_config(vocab_size,
                                    bos_token_id,
                                    eos_token_id))
    if flag_training:
      print("-"*50, "Information of Model", "-"*50)
      print(self.model)
      print("Parameters: ", int(self.model.num_parameters() / 1000000),  "M")
      print("-"*50, "Information of Model", "-"*50)

  def setup_config(self, vocab_size, bos_token_id, eos_token_id):
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    return config

  def create_model(self, config):
    model = GPT2LMHeadModel(config)
    return model