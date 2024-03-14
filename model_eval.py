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
from data_utils import (load_cv_india_dataset, load_sd_qa_test_dataset, filter_data, load_cv_us_dataset)

# os.system("pip install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
# os.system("pip install -q git+https://github.com/huggingface/peft.git@main")

def main():
    wers = []

    model_path = "amyguan/224n-whisper-large-n_ind"
    model = attach_peft(model_path)
    print(f'MODEL LOADED {model_path}')

    print(f'========================MODEL: {model_path}========================')
    print(f'------GETTING CV INDIA DATASET------')
    # CV
    dataset = load_cv_india_dataset() # cv: train = 10%, ~ 900; test = 90%, ~ 9k
    dataset1k = dataset["test"].select(range(1000))

    # SD-QA TEST
    # source = "ind_n"
    # target = "usa"
    # dataset = filter_data(load_sd_qa_test_dataset(), source=source, target=target)
    # dataset['test'] = dataset['test'].rename_column(source, "audio")
    # dataset['test'] = dataset['test'].filter(lambda x: x['question'] is not None)
    # print(dataset)

    print('EVALUATING INDIA')
    # wer = new_evaluate(model, dataset["test"])
    wer = new_evaluate(model, dataset_1k)
    print(f'INDIA WER: {wer}\n')
    wers.append(wer)

    print(f'\n------GETTING CV US DATASET------')
    dataset = load_cv_us_dataset() # cv: train = 10%, ~ 900; test = 90%, ~ 9k
    dataset1k = dataset["test"]

    print('EVALUATING US')
    wer = new_evaluate(model, dataset_1k)
    print(f'US WER: {wer}\n')
    wers.append(wer)

    print(f'\nTOTAL WERS:\n{wers}')


if __name__ == "__main__":
    main()
