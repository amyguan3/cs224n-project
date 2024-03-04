import torch
import os
from transformers import (Seq2SeqTrainer, 
                          TrainingArguments, 
                          TrainerState)
from geomloss import SamplesLoss

class AlignmentSeq2SeqTrainer(Seq2SeqTrainer):
  """
  Trainer with custom earthmover loss
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2)

  def compute_loss(self, model, inputs, return_outputs=False):
    # labels = inputs.get("labels")
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(device)

    # Get first (source) hidden representation
    # print("input features", inputs["input_features"].shape)
    # print("target embeddings", inputs["target_embeddings"].shape)
    output = model(input_features = inputs["input_features"], decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    source_hidden_state = output.encoder_hidden_states[-1]
    # print("source hidden state", source_hidden_state.shape)

    # # Get second (target) hidden_representation
    # input_2 = {"input_features": inputs["input_features"][1:]}  # Remove first element
    # with torch.no_grad():
    #    output_2 = model(**input_2, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    # hidden_state_2 = output_2.encoder_hidden_states[-1]
    target_hidden_state = inputs["target_embeddings"]

    # Get sinkhorn/earthmover loss
    loss = torch.mean(self.sinkhorn_loss(source_hidden_state, target_hidden_state))

    return (loss, output) if return_outputs else loss
  
  def compute_metrics(self, model, eval_data):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(device)
    with torch.no_grad():
      output = model(input_features = eval_data["input_features"], decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    source_hidden_state = output.encoder_hidden_states[-1]
    target_hidden_state = eval_data["target_embeddings"]

    # Get sinkhorn/earthmover loss
    loss = torch.mean(self.sinkhorn_loss(source_hidden_state, target_hidden_state))

    return {'sinkhorn loss': loss}
  
  def evaluate(self, eval_data):
    # change this for common voice WER in future, for now will evaluate on sd_qa test data
    eval_results = self.compute_metrics(self.model, eval_data)
    self.logger.info("Evaluation Results: %s", eval_results)
    return eval_results
