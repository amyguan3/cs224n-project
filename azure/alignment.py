'''
Alignment Whisper embeddings of source input language to target input language
'''

# Import libraries
import logging
import os
import sys
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from huggingface_hub import notebook_login

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
                  LoraModel, 
                  LoraConfig, 
                  TaskType,
                  get_peft_model)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import check_min_version
import re

from trainer_utils import AlignmentSeq2SeqTrainer
from data_utils import DataCollatorSpeechSeq2SeqWithPadding

# Setup 
current = os.path.dirname(os.path.realpath(__file__))  # name of this directory
parent = os.path.dirname(current)  # parent directory
sys.path.append(parent)  # add parent directory to sys.path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use first gpu on machine

logger = logging.getLogger(__name__)
check_min_version("4.21.0")  # calls an error if minimal version of Transformers is not installed. 

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

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def main():
    # log in to huggingface to save model as you go
    notebook_login()

    # load whisper feature extractor, tokenizer, processor
    model_path = "openai/whisper-base"
    task = "transcribe"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_path, task=task)

    target_dialect = 'usa'
    source_dialect = 'ind_n'
    sd_qa = filter_data(load_sd_qa_dataset(), source=source_dialect, target=target_dialect)

    # function to extract audio features (log-Mel input features from audio array)
    def prepare_dataset(batch):
        batch["source_input_features"] = feature_extractor(batch[source_dialect]["array"], sampling_rate=batch[source_dialect]["sampling_rate"]).input_features[0]
        batch["target_input_features"] = feature_extractor(batch[target_dialect]["array"], sampling_rate=batch[target_dialect]["sampling_rate"]).input_features[0]
        return batch
    sd_qa = sd_qa.map(prepare_dataset, remove_columns=[source_dialect, target_dialect], num_proc=2)

    # define an evaluation function !!!

    # data_collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # load pre-trained checkpoint in 8b
    model = WhisperForConditionalGeneration.from_pretrained(model_path,load_in_8bit=True)
    # model.hf_device_map = {" ":0}  # not super sure what to map to here
    model.config.forced_decoder_ids = None  # no tokens forced for decoder outputs
    model.config.suppress_tokens = []

    model = prepare_model_for_int8_training(model)  # freeze all layers and cast non int8 layers to float32

    #----------LORA PART------------
    target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2']
    config = LoraConfig(r=32, # rank, adjust this
                    lora_alpha=64, 
                    target_modules = target_modules, 
                    lora_dropout=0.05, 
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,  # check this???
                    )  
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Define training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir="azure-224n/whisper-base-aligned",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=3,
        evaluation_strategy="steps",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=100,
    #    max_steps=100, # only for testing purposes, remove this from your final run :)
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    trainer = AlignmentSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=sd_qa['dev'],
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    
    trainer.train()







    



if __name__ == "__main__":
    main()