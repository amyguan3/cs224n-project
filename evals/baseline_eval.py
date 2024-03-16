# Import libraries
from transformers import WhisperForConditionalGeneration
from eval_utils import new_evaluate
from data_utils import get_cv_split
import torch

def main():
    # load whisper feature extractor, tokenizer, processor
    model_path = "openai/whisper-large-v2"
    model = WhisperForConditionalGeneration.from_pretrained(model_path, load_in_8bit=True, device_map="auto")

    print('GETTING DATASET')
    dataset = get_cv_split(["India and South Asia (India, Pakistan, Sri Lanka)"])

    print('EVALUATING')
    wer = new_evaluate(model, dataset["train"])
    print(f'BASELINE INDIA WER: {wer}\n')


if __name__ == "__main__":
    main()
