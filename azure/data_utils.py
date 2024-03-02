import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict
import sys

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # convert source inputs to pytorch tensors
        input_features = [{"input_features": [feature["source_input_features"], feature["target_input_features"]]} 
                          for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        return batch

# Functions for loading, processing data
def load_sd_qa_dataset():
    sd_qa = DatasetDict()
    sd_qa["dev"] = load_dataset("WillHeld/SD-QA", split="dev", token=True)
    sd_qa["test"] = load_dataset("WillHeld/SD-QA", split="test", token=True)
    return sd_qa

def filter_data(data, source, target):
    dialect_options = ['aus', 'gbr', 'ind_n', 'ind_s', 'irl', 'kenya', 'nga', 'nzl', 'phl', 'usa', 'zaf']
    if source == 'all':
        print("Error: not yet implemented.")
        sys.exit(1) 
    elif source not in dialect_options or target not in dialect_options:
        print("Error: source or target language not found in dialect options.")
        sys.exit(1) 
    data = data.select_columns(['id', source, target])
    return data

def prepare_target_embeddings(data, model):
    # compute log-Mel input features from target audio array
    batch_size = 128
    target_embeddings = []
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    for i in range(0, len(data["target_input_features"]), batch_size):
        input_features = torch.tensor(data["target_input_features"][i: i + batch_size])
        with torch.no_grad():
            outputs = model(input_features, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        last_hidden_state = outputs.encoder_hidden_states[-1]
        target_embeddings.extend(
            [embedding.flatten() for embedding in last_hidden_state]
        )
    data["target_embeddings"] = target_embeddings
    return data