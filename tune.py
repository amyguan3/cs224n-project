# Import libraries
import logging
import os
import sys

# # Setup : already in alignment_util
# print("Executing pip installs ...")
# os.system("pip3 install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
# os.system("pip3 install -q git+https://github.com/huggingface/peft.git@main")
# print("Print installs done!")

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
from trainer_utils import AlignmentSeq2SeqTrainer
from data_utils import (DataCollatorSpeechSeq2SeqWithPadding, 
                        load_sd_qa_dataset, 
                        filter_data,
                        SDQA_TO_CV)
import matplotlib.pyplot as plt
from eval_utils import (model_pipeline,
                            evaluate_asr_alt,
                            get_cv_split,
                            get_cv_split_mini)
import optuna
import wandb
from alignment_util import (SavePeftCallback,
                            get_embeddings,
                            train_adapter)
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate

current = os.path.dirname(os.path.realpath(__file__))  # name of this directory
parent = os.path.dirname(current)  # parent directory
sys.path.append(parent)  # add parent directory to sys.path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use first gpu on machine

logger = logging.getLogger(__name__)
check_min_version("4.21.0")  # calls an error if minimal version of Transformers is not installed. 


class ParamConfig:
    def __init__(self, learning_rate, batch_size, rank):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rank = rank


"""
pass in accents according to sdqa labeling
ex: source = ["zaf"], or source = ["zaf", "gbr"]
target = "usa"
"""
def tune(sources, target):
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", task="transcribe")

    # for testing purposes
    sd_qa = get_embeddings(sources, target)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    cv_accents = [SDQA_TO_CV[accent] for accent in sources]
    cv_accents.append(SDQA_TO_CV[target])

    eval_dataset = get_cv_split(accents=cv_accents)

    ######################### HYPERPARAMETER TUNING ############################

    def objective(trial):
        print(f'\n=================================TRIAL {trial.number}=================================')

        # Define hyperparameters to optimize
        learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.005)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        rank = trial.suggest_categorical('rank', [32, 64, 128])
        param_config = ParamConfig(learning_rate=learning_rate, batch_size=batch_size, rank=rank)
        print(f'Hyperparameters for trial {trial.number}:\nlearning rate: {learning_rate}, batch size: {batch_size}, rank: {rank}')

        wandb.init(project="large_test", config={"learning_rate": learning_rate, "batch_size": batch_size, "rank": rank})

        peft_model_path = train_adapter(processor, data_collator, sd_qa, param_config)

        # Evaluate the model
        # peft_model_path = "amyguan/large-tune-test"

        peft_config = PeftConfig.from_pretrained(peft_model_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, peft_model_path) # attaches the PEFT module to the Whisper model
        model.config.use_cache = True

        pipe = model_pipeline(model, processor, baseline=False)
        eval_result = evaluate_asr_alt(pipe, eval_dataset["train"], True)
        print(f'metrics: {eval_result}')

        # cumulative WER
        wer = 0
        for accent in eval_result:
            wer += eval_result[accent]['wer']
        wer /= len(eval_result)
        print(f'Average WER: {wer}')
            
        # Log metrics to wandb
        wandb.log({"trial": trial.number, "eval_wer": wer})

        return wer

    # Define Optuna study
    study = optuna.create_study(direction='minimize')
    # can increase number of trials later
    study.optimize(objective, n_trials=3)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # extra stuff below
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"Importances: {importances}")

        params_sorted = list(importances.keys())
        print(f"Sorted params: {params_sorted}")

        # fig = plot_parallel_coordinate(study)
        # os.system(f"mkdir -p tune_plots")
        # fig.savefig('tune_plots/parallel.png')
        # fig = plot_optimization_history(study)
        # fig.savefig('tune_plots/history.png')
    except:
        print('ERROR GENERATING PLOTS')

    # best model??


def main():
    sources = ["ind_n"]
    target = "usa"
    tune(sources, target)


if __name__ == "__main__":
    main()