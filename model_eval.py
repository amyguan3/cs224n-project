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
from data_utils import get_cv_split

# os.system("pip install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
# os.system("pip install -q git+https://github.com/huggingface/peft.git@main")

def main():
    # NOTE: MAYBE SWITCH THE ID
    wers = [14.514391257503464] # model 0, 1: 15.05310143142989, 2: 14.26812374942281
    for i in range(3, 9):
        print(f'\n======================== MODEL NUMBER {i}========================')
        model = attach_peft(f"asyzhou/224n-whisper-large-overnight-{i}")
        print('MODEL LOADED')

        print('GETTING (FILTERED) DATASET')
        dataset = get_cv_split(["India and South Asia (India, Pakistan, Sri Lanka)"]) # pass in cv sources

        print('EVALUATING')
        wer = new_evaluate(model, dataset["train"])
        print(f'MODEL {i} WER: {wer}\n')
        wers.append(wer)

    print(f'TOTAL WERS FOR MODEL 0 to 8: {wers}')


if __name__ == "__main__":
    main()