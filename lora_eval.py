# Import libraries
import os
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
                          TrainerControl,
                          pipeline)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import (prepare_model_for_int8_training,
                  LoraConfig, 
                  PeftModel, 
                  LoraModel, 
                  LoraConfig, 
                  TaskType,
                  get_peft_model,
                  PeftConfig)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import check_min_version
from tqdm import tqdm
import re

from trainer_utils import AlignmentSeq2SeqTrainer
from data_utils import (DataCollatorSpeechSeq2SeqWithPadding, 
                        load_sd_qa_dataset, 
                        filter_data)
from eval_utils import (evaluate_asr,
                        get_mini_cv)
import csv
import pickle
import evaluate

os.system("pip install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
os.system("pip install -q git+https://github.com/huggingface/peft.git@main")

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

    print('LOADING MODEL')
    peft_model_id = "asyzhou/224n-whisper-base-alignment-milestone"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id) # attaches the PEFT module to the Whisper model
    model.config.use_cache = True

    print('GETTING DATASET')
    dataset = get_mini_cv()

    print('EVALUATING')
    metrics = evaluate_asr(model, processor, dataset, True)
    print(metrics)

if __name__ == "__main__":
    main()