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
    commits = [
        "9c3cd22af7f017e1c63a64ecb1fd80c2b5836c42",
        "932c5cb9f9f2b11aea6c4046b88e8e0cd8f33970",
        "65133c7824259c7c1739f15e7b77866a0014b5b2",
        "251677b50aafe6235da287c460ee4d6f29c61a78",
        "36be67ab24b76f8195d90e3193149806705ac502",
        "7800efc575d8bfe9bef61f6e28ffcc346a11a945",
        "a8bcc2484f73081644dd150fb63fe87426dcc91e",
        "71faee385a9698fc5b91b1f0e47f98ce6c3e3ee3",
        "0fe8d33f84c8131c813f74e64c4bce2ab4cea341"
        ]

    wers = []
    i = 0
    for commit in commits:
        print(f'\n======================== MODEL NUMBER {i}========================')
        model = attach_peft(f"asyzhou/224n-whisper-large-overnight-zaf", commit)
        print('MODEL LOADED')

        print('GETTING (FILTERED) DATASET')
        dataset = get_cv_split(["Southern African (South Africa, Zimbabwe, Namibia)"]) # pass in cv sources

        print('EVALUATING')
        wer = new_evaluate(model, dataset["train"])
        print(f'MODEL {i} WER: {wer}\n')
        wers.append(wer)

        i += 1

    print(f'TOTAL WERS FOR MODEL 0 to 8: {wers}')


if __name__ == "__main__":
    main()
