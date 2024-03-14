# Import libraries
from transformers import WhisperForConditionalGeneration
from eval_utils import new_evaluate
from data_utils import (get_cv_split, filter_data, load_cv_india_dataset, load_cv_us_dataset, load_sd_qa_test_dataset)
import torch

def main():
    # load whisper feature extractor, tokenizer, processor
    model_path = "openai/whisper-large-v2"
    model = WhisperForConditionalGeneration.from_pretrained(model_path, load_in_8bit=True, device_map="auto")

    print('=============EVALUATING BASELINE ON WHISPER LARGE=============')

    print(f'------GETTING CV INDIA DATASET------')
    # CV
    dataset = load_cv_india_dataset() # cv: train = 10%, ~ 900; test = 90%, ~ 9k
    dataset1k = dataset["test"].select(range(1000))

    print('EVALUATING')
    wer = new_evaluate(model, dataset1k)
    print(f'BASELINE CV INDIA WER: {wer}\n')

    print(f'\n------GETTING CV US DATASET------')
    dataset = load_cv_us_dataset() # cv: train = 10%, ~ 900; test = 90%, ~ 9k
    dataset1k = dataset["test"]

    print('EVALUATING')
    wer = new_evaluate(model, dataset1k)
    print(f'BASELINE CV US WER: {wer}\n')

    print(f'\n------GETTING CV PHL DATASET------')
    dataset1k = load_dataset("WillHeld/phl_accent_cv", split="train[8%:31%]", token=True)

    print('EVALUATING')
    wer = new_evaluate(model, dataset1k)
    print(f'BASELINE CV PHL WER: {wer}\n')

    print(f'\n------GETTING SD-QA TEST PHL DATASET------')
    source = "phl"
    target = "usa"
    dataset = filter_data(load_sd_qa_test_dataset(), source=source, target=target)
    dataset['test'] = dataset['test'].rename_column(source, "audio")
    dataset['test'] = dataset['test'].filter(lambda x: x['question'] is not None)
    print(dataset)

    print('EVALUATING')
    wer = new_evaluate(model, dataset['test'])
    print(f'BASELINE SD-QA TEST PHL WER: {wer}\n')


if __name__ == "__main__":
    main()
