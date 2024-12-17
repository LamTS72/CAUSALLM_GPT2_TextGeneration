from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from configs.causal_config import ConfigDataset
class CustomDataset():
    def __init__(self,
                 path_dataset=ConfigDataset.PATH_DATASET,
                 revision=ConfigDataset.REVISION,
                 flag_info=True
                ):

        self.raw_data= self.load_custom_dataset(path_dataset)
        self.size = len(self.raw_data["train"]) + len(self.raw_data["validation"])
        if flag_info:
          print("-"*50, "Information of Dataset", "-"*50)
          print(self.raw_data)
          print("-"*50, "Information of Dataset", "-"*50)

    def load_custom_dataset(self, path_dataset):
      data_train_set = load_dataset(path_dataset+"train", split="train")
      data_val_set = load_dataset(path_dataset+"valid", split="validation")
      return DatasetDict(
          {
              "train": data_train_set.shuffle().select(range(500)),
              "validation" : data_val_set.shuffle().select(range(100))
          }
      )

    def __len__(self):
      return self.size

    def __getitem__(self, index):
      dataset = concatenate_datasets((self.raw_data["train"],
                                      self.raw_data["validation"]))

      return dataset[index]

