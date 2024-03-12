import torch
import os
from transformers import (Seq2SeqTrainer, 
                          TrainingArguments, 
                          TrainerState)
from geomloss import SamplesLoss
import sys
import evaluate
from eval_utils import new_evaluate

class AlignmentSeq2SeqTrainer(Seq2SeqTrainer):
  """
  Trainer with custom earthmover loss
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2)
  
  def evaluate(self, eval_dataset=None, ignore_keys= None, metric_key_prefix='eval'):
    print("Evaluating....")
    wer_metric = evaluate.load("wer")
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    eval_dataloader = self.get_eval_dataloader(eval_dataset)

    eval_predictions = super().evaluate(eval_dataset, metric_for_compute=wer_metric)
    predictions, labels = eval_predictions.predictions, eval_predictions.label_ids
    print(predictions)
    print(labels)
    print(len(predictions))
    print(labels.shape)
    
    wer = 100 * wer_metric.compute(predictions=predictions, references=labels)
    print(wer)
    sys.exit()
    return wer
    


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
