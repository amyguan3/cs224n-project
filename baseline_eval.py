# Import libraries
from transformers import WhisperForConditionalGeneration
from eval_utils import new_evaluate
from data_utils import load_cv_india_dataset
import torch

def main():
    # load whisper feature extractor, tokenizer, processor
    model_path = "openai/whisper-large-v2"
    model = WhisperForConditionalGeneration.from_pretrained(model_path, load_in_8bit=True, device_map="auto")

    print('GETTING DATASET')
    dataset = load_cv_india_dataset()

    print('EVALUATING')
    wer = new_evaluate(model, dataset["train"])
    print(f'BASELINE INDIA WER: {wer}\n')


if __name__ == "__main__":
    main()
