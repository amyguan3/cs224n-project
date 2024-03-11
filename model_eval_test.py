# Import libraries
import os
from huggingface_hub import notebook_login
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration)
import torch
from peft import (PeftModel, 
                  PeftConfig)
from eval_utils_new import (new_evaluate,
                            attach_peft)
from data_utils import get_cv_split

# os.system("pip install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
# os.system("pip install -q git+https://github.com/huggingface/peft.git@main")

def main():
    # NOTE: MAYBE SWITCH THE ID
    model = attach_peft("asyzhou/224n-whisper-large-overnight-0")
    print('MODEL LOADED')

    print('GETTING (FILTERED) DATASET')
    dataset = get_cv_split(["India and South Asia (India, Pakistan, Sri Lanka)"]) # pass in cv sources

    print('EVALUATING')
    metrics = new_evaluate(model, dataset["train"])
    print(metrics)


if __name__ == "__main__":
    main()