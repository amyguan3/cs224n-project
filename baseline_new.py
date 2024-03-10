# Import libraries
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration)
from eval_utils_new import (model_pipeline,
                            evaluate_asr_alt,
                            get_cv_split)

def main():
    # load whisper feature extractor, tokenizer, processor
    model_path = "openai/whisper-large-v2"
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path, language="English", task="transcribe")

    pipe = model_pipeline(model, processor, verbose=True)

    print('GETTING DATASET')
    dataset = get_cv_split() # .to(device)

    print('EVALUATING')
    metrics = evaluate_asr_alt(pipe, dataset["train"], baseline=True, verbose=True)
    print(f'METRICS:\n{metrics}')

if __name__ == "__main__":
    main()