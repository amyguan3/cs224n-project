# Import libraries
import os
from huggingface_hub import notebook_login
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration)
import torch
from peft import (PeftModel, 
                  PeftConfig)
from eval_utils import (new_evaluate,
                            attach_peft)
from data_utils import load_cv_india_dataset

# os.system("pip install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
# os.system("pip install -q git+https://github.com/huggingface/peft.git@main")

def main():
    wers = []

    i = 0
    # for commit in commits:
    print(f'\n======================== MODEL NUMBER {i}========================')
    model = attach_peft(f"amyguan/224n-whisper-large-n_ind")
    print('MODEL LOADED')

    print('GETTING (FILTERED) DATASET')
    dataset = load_cv_india_dataset() # pass in cv sources

    print('EVALUATING')
    wer = new_evaluate(model, dataset["train"])
    print(f'MODEL {i} WER: {wer}\n')
    wers.append(wer)

    i += 1

    print(f'TOTAL WER: {wers}')


if __name__ == "__main__":
    main()
