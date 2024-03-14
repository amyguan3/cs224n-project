import torch
import os
from transformers import (Seq2SeqTrainer, 
                          TrainingArguments, 
                          TrainerState)
from geomloss import SamplesLoss
import sys
import evaluate
from eval_utils import new_evaluate
import numpy as np
import gc
import time

class AlignmentSeq2SeqTrainer(Seq2SeqTrainer):
  """
  Trainer with custom earthmover loss
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2)
  
  def evaluate(self, eval_dataset=None, ignore_keys= None, metric_key_prefix: str = "eval"):
    # print("Evaluating....")
    metric = evaluate.load("wer")
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    # sampled_dataset = eval_dataset.train_test_split(test_size=0.3, seed=42)
    # 30%, ~300 samples

    eval_dataloader = self.get_eval_dataloader(eval_dataset)
    self.model.eval()
    predictions = []
    references = []

    start_time = time.time()
    # super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    for batch in eval_dataloader:
      with torch.cuda.amp.autocast():
        with torch.no_grad(): 
          # inputs = self._prepare_inputs(batch)
          generated_tokens = (self.model.generate(
             input_features=batch["input_features"], max_new_tokens=255)
             )
          labels = batch["labels"]
          #labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id) #unnecessary
          decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
          decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

          predictions.extend(decoded_preds)
          references.extend(decoded_labels)
          del generated_tokens, labels, batch
        gc.collect()

    end_time = time.time()
    eval_time = end_time - start_time

    wer = 100 * metric.compute(predictions=predictions, references=references)
    self.log({f"{metric_key_prefix}_wer": wer, f"{metric_key_prefix}_time": eval_time})

    return {f"{metric_key_prefix}_wer": wer}    

  def compute_loss(self, model, inputs, return_outputs=False):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(device)

    # Get first (source) hidden representation
    output = model(input_features = inputs["input_features"], decoder_input_ids=decoder_input_ids)
    source_hidden_state = output.encoder_last_hidden_state
    # # Get second (target) hidden_representation
    target_hidden_state = inputs["target_embeddings"]

    # Get sinkhorn/earthmover loss
    loss = torch.mean(self.sinkhorn_loss(source_hidden_state, target_hidden_state))
    return (loss, output) if return_outputs else loss
