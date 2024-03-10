'''
Alignment Whisper embeddings of source input language to target input language
'''

# Import libraries
import logging
import os
import sys

# Setup 
os.system("pip install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
os.system("pip install -q git+https://github.com/huggingface/peft.git@main")

import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (WhisperFeatureExtractor, 
                          WhisperTokenizer, 
                          WhisperProcessor,
                          WhisperModel,
                          WhisperForConditionalGeneration, 
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer, 
                          TrainerCallback, 
                          TrainingArguments, 
                          TrainerState, 
                          TrainerControl)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import (prepare_model_for_int8_training,
                  LoraConfig, 
                  PeftModel,
                  PeftConfig, 
                  LoraModel, 
                  LoraConfig, 
                  TaskType,
                  get_peft_model)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import check_min_version
from tqdm import tqdm
import re

from trainer_utils import AlignmentSeq2SeqTrainer
from data_utils import (DataCollatorSpeechSeq2SeqWithPadding, 
                        load_sd_qa_dataset, 
                        filter_data)

import csv
import matplotlib.pyplot as plt
from amy.old.eval_utils import (evaluate_asr,
                    get_mini_cv)
import wandb
import pprint

current = os.path.dirname(os.path.realpath(__file__))  # name of this directory
parent = os.path.dirname(current)  # parent directory
sys.path.append(parent)  # add parent directory to sys.path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use first gpu on machine

logger = logging.getLogger(__name__)
check_min_version("4.21.0")  # calls an error if minimal version of Transformers is not installed. 

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})     






def main():
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # wandb.login()

    sweep_config = {
        'method': 'random'
        }

    metric = {
        'name': 'wer',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric

    # TODO: come back to this
    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'weight_decay': {
            'values': [0.0, 0.1, 0.2]
            },
        }
    
    sweep_config['parameters'] = parameters_dict

    # set permanent hyperparameters
    parameters_dict.update({
    'epochs': {
        'value': 3}
    })

    # random search distributions
    parameters_dict.update({
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 8,
            'max': 64,
        }
        })

    pprint.pprint(sweep_config)
    
    # look at other config options, like early stopping

    sweep_id = wandb.sweep(sweep_config, project="base-test")



if __name__ == "__main__":
    main()
