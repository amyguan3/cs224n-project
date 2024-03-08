import torch
import os
from transformers import (Seq2SeqTrainer, 
                          TrainingArguments, 
                          TrainerState)
from geomloss import SamplesLoss
import sys

class AlignmentSeq2SeqTrainer(Seq2SeqTrainer):
  """
  Trainer with custom earthmover loss
  """
  def __init__(self, embedding_save_folder, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2)
    self.embedding_save_folder = embedding_save_folder

  def compute_loss(self, model, inputs, return_outputs=False):
    print(input.keys())
    sys.exit()
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(device)

    # Get first (source) hidden representation
    output = model(input_features = inputs["input_features"], decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    source_hidden_state = output.encoder_hidden_states[-1]

    # Get second (target) hidden_representation
    # target_hidden_state = inputs["target_embeddings"]
    target_hidden_state = []
    for id in inputs["id"]:
      embedding_load_path =  f"{self.embedding_save_folder}/{id}.pt"
      target = torch.load(embedding_load_path)
      target_hidden_state.append(target)
    target_hidden_state = torch.cat(target_hidden_state)

    # Get sinkhorn/earthmover loss
    loss = torch.mean(self.sinkhorn_loss(source_hidden_state, target_hidden_state))
    return (loss, output) if return_outputs else loss
