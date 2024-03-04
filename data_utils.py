import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict
import sys

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, data: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # convert source inputs to pytorch tensors
        print(len(data))
        input_features = [{"input_features": example["source_input_features"]} for example in data]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # format target embeddings
        target_embeddings = torch.cat([torch.tensor(example["target_embeddings"]) for example in data], axis = 0)
        batch["target_embeddings"] = target_embeddings
        print(target_embeddings.shape)

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
