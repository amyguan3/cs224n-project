import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Dataset
import sys
import pandas as pd

SDQA_TO_CV = {
              "phl": "Filipino",
              "usa": "United States English",
              "gbr": "Scottish English",
              "zaf": "Southern African (South Africa, Zimbabwe, Namibia)",
              "ind_n": "India and South Asia (India, Pakistan, Sri Lanka)",
              "aus": "Australian English",
              "nzl": "New Zealand English",
              "irl": "Irish English",
              "kenya": "Kenyan English",
              "nga": "nigeria english"
              }

CV_ACCENTS = ["Filipino",
           "United States English",
           "Scottish English",
           "Southern African (South Africa, Zimbabwe, Namibia)",
           "Hong Kong English",
           "India and South Asia (India, Pakistan, Sri Lanka)",
           "Australian English",
           "New Zealand English",
           "Irish English",
           "Kenyan English",
           "nigeria english"
           ]

CV_ACCENTS_SDQA = ["Filipino",
           "United States English",
           "Southern African (South Africa, Zimbabwe, Namibia)",
           "India and South Asia (India, Pakistan, Sri Lanka)",
           "Irish English"
           ]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, data: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # convert source inputs to pytorch tensors
        input_features = [{"input_features": example["source_input_features"]} for example in data]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # format target embeddings # training
        if "target_embeddings" in data[0]:
            target_embeddings = torch.cat([torch.tensor(example["target_embeddings"]) for example in data], axis = 0)
            batch["target_embeddings"] = target_embeddings

        # format labels # inference
        if "labels" in data[0]:
            label_features = [{"input_ids": example["labels"]} for example in data]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            # replace padding with -100 to ignore loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            # check beginning-of-sequence-token
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels

        return batch


# Functions for loading, processing data
def load_sd_qa_dataset():
    sd_qa = DatasetDict()
    sd_qa["dev"] = load_dataset("WillHeld/SD-QA", split="dev", token=True)
    # sd_qa["test"] = {} # load_dataset("WillHeld/SD-QA", split="test", token=True)
    return sd_qa

# Functions for loading, processing data
def load_sd_qa_test_dataset():
    sd_qa = DatasetDict()
    sd_qa["test"] = load_dataset("WillHeld/SD-QA", split="test", token=True)
    return sd_qa

def load_cv_india_dataset():
    # cv = DatasetDict()
    cv = load_dataset("WillHeld/india_accent_cv", split='train[:1%]', token=True)
    cv = cv.train_test_split(test_size=0.3, seed=42)
    return cv


def load_cv_phl_dataset():
    cv = load_dataset("WillHeld/phl_accent_cv", split='train[:25%]', token=True)
    cv = cv.train_test_split(test_size=0.3, seed=42)
    return cv


"""
note: casts source dialect col to "audio" (including in 1-1 case)
"""
def filter_data(data, source, target):
    # dialect_options = ['aus', 'gbr', 'ind_n', 'ind_s', 'irl', 'kenya', 'nga', 'nzl', 'phl', 'usa', 'zaf']
    dialect_options = ['ind_n', 'irl', 'phl', 'usa', 'zaf']
    if source == 'all':
        # explode across source dialects
        dialect_options.remove(target)
        df = pd.DataFrame(data['dev'])
        df = pd.melt(df, id_vars=['id', 'question', target], value_vars=dialect_options, var_name='accent', value_name="all")
        return Dataset.from_pandas(df)
    elif source not in dialect_options or target not in dialect_options:
        print("Error: source or target language not found in dialect options.")
        sys.exit(1) 
    else:
        data = data.select_columns(['id', 'question', source, target])
        return data


"""
returns cv with
cv["train"]
cv["test]
50% each
"""
def get_cv_split(accents=CV_ACCENTS):
    print("[get_cv_split] Loading CV dataset...")
    cv_all = load_dataset("WillHeld/accented_common_voice", split="train", token=True, revision="e5b7f595177ccdb4a599f3589ce01957b0330357")
    cv_all = cv_all.shuffle(seed=42)
    cv_all = cv_all.select(range(10_000))

    # data split
    cv_split = cv_all.train_test_split(test_size=0.5, seed=42)
    cv_split = cv_split.filter(lambda example: example['accents'] in accents)

    print("[get_cv_split] CV dataset loaded!")

    return cv_split


def get_cv_split_mini(accents=CV_ACCENTS):
    print("LOADING MINI CV DATASET")
    cv_all = load_dataset("WillHeld/accented_common_voice", split="train", token=True, revision="e5b7f595177ccdb4a599f3589ce01957b0330357")
    cv_all = cv_all.shuffle(seed=42)
    cv_all = cv_all.select(range(32))

    # data split
    cv_split = cv_all.train_test_split(test_size=0.5, seed=42)
    cv_split = cv_split.filter(lambda example: example['accents'] in accents)

    print("MINI CV DATASET LOADED")
    return cv_split


def get_counts():
    cv = get_cv_split()

    # val = pd.DataFrame(cv["train"]["accents"])
    test = pd.DataFrame(cv["test"]["accents"])
    pd.set_option('display.max_rows', None)
    print(test.value_counts())
