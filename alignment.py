'''
Alignment Whisper embeddings of source input language to target input language
'''

# Import libraries
import logging
import os
import sys

# Setup 
print("Executing pip installs ...")
os.system("pip3 install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
os.system("pip3 install -q git+https://github.com/huggingface/peft.git@main")
print("Print installs done!")

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
from eval_utils import (evaluate_asr,
                    get_mini_cv)


current = os.path.dirname(os.path.realpath(__file__))  # name of this directory
parent = os.path.dirname(current)  # parent directory
sys.path.append(parent)  # add parent directory to sys.path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use first gpu on machine

logger = logging.getLogger(__name__)
check_min_version("4.21.0")  # calls an error if minimal version of Transformers is not installed. 



class SavePeftCallback(TrainerCallback):
    def __init__(self):
        self.training_losses =[]
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if "loss" in state.log_history[-1].keys():
            self.training_losses.append([state.global_step, state.log_history[-1]["loss"]])

    
    def plot_loss(self):
        plt.plot(self.training_losses)
        plt.xlabel("training step")
        plt.ylabel("loss")
        plt.title("loss over training steps")
        plt.savefig("loss_output_plot.png")

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
    # log in to huggingface with huggingface-cli login
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print("Torch cuda is available?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Torch cuda current device?", torch.cuda.current_device())

    #------------------------------------#
    #---------------MODEL----------------#
    #------------------------------------#
    print("Loading model...")

    # load whisper feature extractor, tokenizer, processor
    # model_path = "openai/whisper-base"
    model_path = "openai/whisper-large-v2"
    task = "transcribe"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_path, task=task)

    # load pre-trained model checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.forced_decoder_ids = None  # possibly this needs editing
    model.config.suppress_tokens = []
    model = model.to(device)  # doesn't work with 8bit
    print("Model probably saved to device:", device)

    #------------------------------------#
    #----------------DATA----------------#
    #------------------------------------#
    print("Loading data...")

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
        
        # encode question text to label ids
        # data["labels"] = tokenizer(data[source_dialect]["question"]).input_ids
        return data

    # prepare targets
    def prepare_target_embeddings(data):
        # compute encoder embedding from target audio array
        decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(device)
        input_features = torch.tensor(data["target_input_features"]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_features, decoder_input_ids=decoder_input_ids)
        last_hidden_state = outputs.encoder_last_hidden_state
        data["target_embeddings"] = [embedding for embedding in last_hidden_state]
        return data
    
    sd_qa = sd_qa.map(prepare_source_data, desc="Extract features for source dialect"
                      ).map(prepare_target_embeddings, desc="Original hidden embeddings for target dialect")


    # define an evaluation function !!!
    # metric = evaluate.load("wer")

    print(sd_qa)
    sd_qa.remove_columns('id')
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    #------------------------------------#
    #--------------TRAINING--------------#
    #------------------------------------#

    print("Load 8bit model...")
    print("Start training...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
    model.config.forced_decoder_ids = None  # possibly this needs editing
    model.config.suppress_tokens = []
    model = prepare_model_for_int8_training(model)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)


    target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2']
    config = LoraConfig(r=32, # rank, adjust this
                    lora_alpha=64, 
                    target_modules = target_modules, 
                    lora_dropout=0.05, 
                    bias="none",
                    # task_type=TaskType.FEATURE_EXTRACTION,  # check this???
                    )  
    model = get_peft_model(model, config)
    # model.config.gradient_checkpointing = True

    model.print_trainable_parameters()

    # Define training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir="model_checkpoints",  
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        # gradient_checkpointing=True, # just added
        num_train_epochs=3,
        # evaluation_strategy="steps",  # disregard since using commonvoice to eval
        # per_device_eval_batch_size=8,
        fp16=True,  # don't think we need this
        generation_max_length=128,
        logging_steps=20,
        remove_unused_columns=False, 
    )
    peftcallback = SavePeftCallback()
    trainer = AlignmentSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=sd_qa['dev'],
        eval_dataset=sd_qa['dev'],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[peftcallback],
    )

    trainer.train()
    # peft_model_id = "asyzhou/224n-whisper-base-alignment-milestone"
    print("Done with training! Pushing to hub...")
    peft_model_id = "asyzhou/224n-whisper-large-alignment-test"
    model.push_to_hub(peft_model_id)
    peftcallback.plot_loss()

    # peft_config = PeftConfig.from_pretrained(peft_model_id)
    # model = WhisperForConditionalGeneration.from_pretrained(
    #     peft_config.base_model_name_or_path, device_map="auto"
    # )
    # model = PeftModel.from_pretrained(model, peft_model_id) # attaches the PEFT module to the Whisper model
    # model.config.use_cache = True

    # dataset = get_mini_cv() # .to(device)
    # metrics = evaluate_asr(model, processor, dataset, True)
    # print(metrics)

    

if __name__ == "__main__":
    main()
