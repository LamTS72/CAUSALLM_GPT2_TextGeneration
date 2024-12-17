from transformers import (
    get_scheduler,
)
# import evaluate
import torch
import os
import numpy as np
# import evaluate
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import Repository, HfApi, HfFolder
import math
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from configs.causal_config import ConfigModel, ConfigHelper
from causal_model import CustomModel
from preprocessing import Preprocessing
from data.custom_data import CustomDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Used Device: ", device)

class Training():
    def __init__(self, model_name=ConfigModel.MODEL_NAME,
                 learning_rate=ConfigModel.LEARNING_RATE,
                 epoch=ConfigModel.EPOCHS,
                 num_warmup_steps=ConfigModel.NUM_WARMUP_STEPS,
                 name_metric=ConfigModel.METRICs,
                 path_tensorboard=ConfigModel.PATH_TENSORBOARD,
                 path_save=ConfigModel.PATH_SAVE,
                 gradient_accumulation_steps = ConfigModel.GRADIENT_ACCUMULATION_STEPS,
                 eval_steps = ConfigModel.EVAL_STEPS,
                 weight_decay = ConfigModel.WEIGHT_DECAY,
                 dataset=None,
                 process=None
                ):
        self.dataset = dataset
        self.process = process
        self.model = CustomModel(vocab_size=len(self.process.tokenizer),
                                 bos_token_id=self.process.tokenizer.bos_token_id,
                                 eos_token_id=self.process.tokenizer.eos_token_id).model

        self.epochs = epoch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.weight_decay = weight_decay
        self.keytoken_ids = self.keytoken_ids()
        self.num_steps = self.epochs * len(self.process.train_loader)
        self.optimizer = torch.optim.AdamW(
            self.get_grouped_params(self.model),
            lr=learning_rate
        )
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_steps
        )
        #self.metric = evaluate.load(name_metric)
        self.writer = SummaryWriter(path_tensorboard)

        # Define necessary variables
        self.api = HfApi(token=ConfigHelper.TOKEN_HF)
        self.repo_name = path_save  # Replace with your repo name
        self.author = ConfigHelper.AUTHOR
        self.repo_id = self.author + "/" + self.repo_name
        self.token = HfFolder.get_token()
        self.repo = self.setup_hf_repo(self.repo_name, self.repo_id, self.token)

    def setup_hf_repo(self, local_dir, repo_id, token):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        try:
            self.api.repo_info(repo_id)
            print(f"Repository {repo_id} exists. Cloning...")
        except Exception as e:
            print(f"Repository {repo_id} does not exist. Creating...")
            self.api.create_repo(repo_id=repo_id, token=token, private=True)

        repo = Repository(local_dir=local_dir, clone_from=repo_id)
        return repo

    def save_and_upload(self, epoch, final_commit=False):
        # Save model, tokenizer, and additional files
        self.model.save_pretrained(self.repo_name)
        self.process.tokenizer.save_pretrained(self.repo_name)

        # Push to Hugging Face Hub
        self.repo.git_add(pattern=".")
        commit_message = "Final Commit: Complete fine-tuned model" if final_commit else f"Epoch {epoch}: Update fine-tuned model and metrics"
        self.repo.git_commit(commit_message)
        self.repo.git_push()

        print(f"Model and files pushed to Hugging Face Hub for epoch {epoch}: {self.repo_id}")

    def keytoken_weighted_loss(self, inputs, logits, keytoken_ids, alpha=1.0):
      # Dịch chuyển để token < n dự đoán n
      shift_labels = inputs[..., 1:].contiguous()
      shift_logits = logits[..., :-1, :].contiguous()
      # Tính độ mất mát từng token
      loss_fct = CrossEntropyLoss(reduce=False)
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      # Thay đổi kích thước và mất mát trung bình trên mỗi mẫu
      loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
      # Tính toán và chia tỷ trọng
      weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
          axis=[0, 2]
      )
      weights = alpha * (1.0 + weights)
      # Tính giá trị trung bình có trọng số
      weighted_loss = (loss_per_sample * weights).mean()
      return weighted_loss

    def get_grouped_params(self, model, no_decay=["bias", "LayerNorm.weight"]):
      params_with_wd, params_without_wd = [], []
      for n, p in model.named_parameters():
          if any(nd in n for nd in no_decay):
              params_without_wd.append(p)
          else:
              params_with_wd.append(p)
      return [
          {"params": params_with_wd, "weight_decay": self.weight_decay},
          {"params": params_without_wd, "weight_decay": 0.0},
      ]

    def evaluate(self):
      self.model.eval()
      losses = []
      for step, batch in enumerate(self.process.val_loader):
          with torch.no_grad():
              outputs = self.model(batch["input_ids"], labels=batch["input_ids"])

          losses.append(outputs.loss)
      loss = torch.mean(torch.cat(losses))
      try:
          perplexity = torch.exp(loss)
      except OverflowError:
          perplexity = float("inf")
      return loss.item(), perplexity.item()

    def keytoken_ids(self):
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
                
    def fit(self, flag_step=False):
      completed_steps = 0
      for epoch in range(self.epochs):
        for step, batch in tqdm(enumerate(self.process.train_loader, start=1), total=self.num_steps):

            logits = self.model(batch["input_ids"]).logits
            loss = self.keytoken_weighted_loss(batch["input_ids"], logits, self.keytoken_ids)
            if step % 100 == 0:
                print(
                    {
                        "lr": self.lr_scheduler.get_lr(),
                        "steps": completed_steps,
                        "loss/train": loss.item() * self.gradient_accumulation_steps,
                    }
                )
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            if step % self.gradient_accumulation_steps == 0:
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                completed_steps += 1
            if (step % (self.eval_steps * self.gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = self.evaluate()
                print({"loss/eval": eval_loss, "perplexity": perplexity})
        # Save and upload after each epoch
        final_commit = ((epoch+1) == self.epochs)
        self.save_and_upload((epoch+1), final_commit)


if __name__ == '__main__':
    dataset = CustomDataset()
    process = Preprocessing(dataset=dataset.raw_data, local_processed=False)
    train = Training(dataset=dataset,process=process)
    train.fit()
