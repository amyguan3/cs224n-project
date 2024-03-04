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
from tqdm import tqdm
import re

from trainer_utils import AlignmentSeq2SeqTrainer
from data_utils import (DataCollatorSpeechSeq2SeqWithPadding, 
                        load_sd_qa_dataset, 
                        filter_data)

import csv


current = os.path.dirname(os.path.realpath(__file__))  # name of this directory
parent = os.path.dirname(current)  # parent directory
sys.path.append(parent)  # add parent directory to sys.path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use first gpu on machine

logger = logging.getLogger(__name__)
check_min_version("4.21.0")  # calls an error if minimal version of Transformers is not installed. 



class SavePeftCallback(TrainerCallback):
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

        # record the losses
        loss_file = os.path.join(args.output_dir, 'loss.csv')
        with open(loss_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([state.global_step, state.log_history["loss"][-1]]) # iter, loss

        return control


def main():
    # log in to huggingface to save model as you go
    # notebook_login()
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # load whisper feature extractor, tokenizer, processor
    model_path = "openai/whisper-base"
    task = "transcribe"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_path, task=task)

    # load pre-trained model checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    # model = prepare_model_for_int8_training(model)
    # model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
    model.config.forced_decoder_ids = None  # no tokens forced for decoder outputs
    model.config.suppress_tokens = []
    model = model.to(device)
    # def make_inputs_require_grad(module, input, output):
    #     output.requires_grad_(True)
    # model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    
    # load data
    target_dialect = 'gbr'
    source_dialect = 'ind_n'
    sd_qa = filter_data(load_sd_qa_dataset(), source=source_dialect, target=target_dialect)
    
    print(sd_qa['dev'][0])

    # prepare data
    def prepare_source_data(data):
        # compute log-Mel input features from audio arrays
        data["source_input_features"] = feature_extractor(data[source_dialect]["array"], sampling_rate=data[source_dialect]["sampling_rate"]).input_features[0]
        data["target_input_features"] = feature_extractor(data[target_dialect]["array"], sampling_rate=data[target_dialect]["sampling_rate"]).input_features[0]
        return data

    # prepare targets
    def prepare_target_embeddings(data):
        # compute encoder embedding from target audio array
        target_embeddings = []
        decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(device)
        # for i in range(0, len(data["target_input_features"]), batch_size):
        input_features = torch.tensor(data["target_input_features"]).unsqueeze(0).to(device)
        # print(input_features.shape)
        with torch.no_grad():
            outputs = model(input_features, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        last_hidden_state = outputs.encoder_hidden_states[-1]
        target_embeddings = [embedding for embedding in last_hidden_state]
        data["target_embeddings"] = target_embeddings
        return data
    
    sd_qa = sd_qa.map(prepare_source_data, desc="Extract features for source dialect"
                      ).map(prepare_target_embeddings, desc="Original hidden embeddings for target dialect")


    # define an evaluation function !!!
    
    print(sd_qa)
    # data_collator
    sd_qa.remove_columns('id')
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    #----------LORA PART------------
    target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2']
    config = LoraConfig(r=32, # rank, adjust this
                    lora_alpha=64, 
                    target_modules = target_modules, 
                    lora_dropout=0.05, 
                    bias="none",
                    # task_type=TaskType.FEATURE_EXTRACTION,  # check this???
                    )  
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Define training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir="azure-224n/test",  
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
        # max_steps=100, # only for testing purposes, remove this from your final run :)
        remove_unused_columns=False, 
    )

    trainer = AlignmentSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=sd_qa['dev'],
        eval_dataset=sd_qa['test'],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftCallback],
    )

    trainer.train()
    peft_model_id = "azure-224n/whisper-base-alignment"
    model.push_to_hub(peft_model_id)

    

if __name__ == "__main__":
    main()
