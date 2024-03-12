import torch
import os
from transformers import (Seq2SeqTrainer, 
                          TrainingArguments, 
                          TrainerState)
from geomloss import SamplesLoss
import sys
from eval_utils import new_evaluate

class AlignmentSeq2SeqTrainer(Seq2SeqTrainer):
  """
  Trainer with custom earthmover loss
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2)
  
  def evaluate(self, eval_dataset, ignore_keys= None, metric_key_prefix='eval'):
    if eval_dataset:
      raise ValueError("Need an eval_dataset to evaluate.")
    eval_predictions = super().evaluate(eval_dataset["train"])
    print(eval_predictions)
    sys.exit()
    wer = new_evaluate(self.model, eval_dataset["train"], one_to_one=True)
    


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
