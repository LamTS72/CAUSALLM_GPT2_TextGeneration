from transformers import(
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from configs.causal_config import ConfigModel

class Preprocessing():
    def __init__(self, model_tokenizer=ConfigModel.MODEL_TOKENIZER,
                 batch_size=ConfigModel.BATCH_SIZE,
                 chunk_size=ConfigModel.CHUNK_SIZE,
                 train_size=ConfigModel.TRAIN_SIZE,
                 ratio=ConfigModel.RATIO,
                 context_length=ConfigModel.CONTEXT_LENGTH,
                 dataset=None,
                 flag_training=True,
                 local_processed=False):
      self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.context_length = context_length
      self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                           mlm=False)
      if flag_training:
        print("-"*50, "Information of Tokenizer", "-"*50)
        print(self.tokenizer)

        print("-"*50, "Information of Tokenizer", "-"*50)
        self.chunk_size = chunk_size
        if local_processed:
          self.tokenized_dataset = load_from_disk("/kaggle/working/data")
        else:
          self.tokenized_dataset = self.map_tokenize_dataset(dataset)

        print(self.tokenized_dataset)
        self.train_loader, self.val_loader = self.data_loader(batch_size)
        self.keytoken_ids = self.key_token_ids()
    def key_token_ids(self):
      keytoken_ids = []
      for keyword in [
          "plt",
          "pd",
          "sk",
          "fit",
          "predict",
          " plt",
          " pd",
          " sk",
          " fit",
          " predict",
          "testtest",
      ]:
          ids = self.process.tokenizer([keyword]).input_ids[0]
          if len(ids) == 1:
              keytoken_ids.append(ids[0])
          else:
              print(f"Keyword has not single token: {keyword}")
          return keytoken_ids

    def tokenize_dataset(self, sample):
      tokenized_input = self.tokenizer(
          sample["content"],
          truncation=True,
          max_length=self.context_length,
          return_overflowing_tokens=True,
          return_length=True
      )
      input_batch = []
      for length, input_ids in zip(tokenized_input["length"], tokenized_input["input_ids"]):
        if length == self.context_length:
          input_batch.append(input_ids)
      return {"input_ids": input_batch}

    def map_tokenize_dataset(self, dataset):
      tokenized_dataset = dataset.map(
          self.tokenize_dataset,
          batched=True,
          remove_columns=dataset["train"].column_names
      )
      tokenized_dataset.save_to_disk("/kaggle/working/data")
      return tokenized_dataset

    def data_loader(self, batch_size):
      self.tokenized_dataset.set_format("torch")
      train_loader = DataLoader(
        self.tokenized_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=self.data_collator,
      )
      val_loader = DataLoader(
        self.tokenized_dataset["validation"],
        batch_size=batch_size,
        collate_fn=self.data_collator
      )
      return train_loader, val_loader

