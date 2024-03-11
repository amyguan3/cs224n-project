from huggingface_hub import interpreter_login
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, pipeline
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm import tqdm
import os
import torch
import pickle
import numpy as np
from data_utils import DataCollatorSpeechSeq2SeqWithPadding
from torch.utils.data import DataLoader
import gc
from peft import (PeftModel,
                  PeftConfig)
from collections import defaultdict

"""
OLD VERSION BELOW THAT I ORIGINALLY WROTE FOR ITERABLE DATASET
"""
def get_text(sample):
    # can replace with just return sample["sentence"]?
    if "sentence" in sample:
        return sample["sentence"]
    elif "question" in sample:
        return sample["question"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )
    

def get_accents(sample):
    if "accents" in sample:
        return sample["accents"]
    elif "accent" in sample:
        return sample["accent"]
    else:
        raise ValueError(
            f"Expected transcript column of accent. Ensure an accent column is present in the dataset."
        )


def data(dataset):
    # MODIFY THIS FOR SD-QA SINCE GET_ACCENTS WON'T WORK
    for i, item in enumerate(dataset):
        yield {"raw": np.asarray(item["audio"]["array"]), "sampling_rate": item["audio"]["sampling_rate"], "reference": get_text(item), "accents": get_accents(item)}


def model_pipeline(model, processor, baseline=False, verbose=True):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    if baseline == False: # ADAPTER VER, USES ACCELERATOR
        whisper_asr = pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor
        ) # , device=device
    else:
        whisper_asr = pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=device
        )
    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
    )
    whisper_asr.model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )
    if verbose:
        print("MODEL PIPELINE SET UP")
    return whisper_asr


def get_preds(asr_model, dataset_total, verbose):
    predictions = {}
    references = {}
    all_accents = []

    i = 0

    with torch.cuda.amp.autocast():
        for out in tqdm(asr_model(data(dataset_total), batch_size=4), desc='Decode Progress'):
            accent = out["accents"][0]
            if accent not in all_accents:
                all_accents.append(accent)
                predictions[accent] = []
                references[accent] = []

            predictions[accent].append(out["text"])
            references[accent].append(out["reference"][0])

            i += 1
            if i % 100 == 0:
                if verbose:
                    print(f'\niteration: {i}')

    return predictions, references, all_accents


def save_metrics(metrics, references, predictions, accent, wer):
    acc_name = whisper_norm(accent).strip().replace(' ', '_')

    metrics[acc_name] = {'wer': wer}

    os.system(f"mkdir -p evaluation")
    op_file = f"evaluation/{acc_name}.txt"
    result_file = open(op_file, 'w')
    result_file.write('ACCENT: ' + str(accent) + '\n')
    result_file.write('\nWER: ' + str(wer) + '\n')

    for ref, pred in zip(references[accent], predictions[accent]):
        result_file.write(f"REF: {ref.encode('utf-8')}\n")
        result_file.write(f"PRED: {pred.encode('utf-8')}\n")
        result_file.write("------------------------------------------------------" + '\n')
    result_file.close()


def evaluate_asr_alt(whisper_asr, dataset, verbose=True):
    wer_metric = evaluate.load("wer")

    # RUN INFERENCE
    predictions, references, all_accents = get_preds(whisper_asr, dataset, verbose)

    if verbose:
        print('\n ASR COMPLETED\n')

    # COMPUTE METRICS
    wer_metric = evaluate.load("wer")
    metrics = {}

    for accent in all_accents:
        wer = 100 * wer_metric.compute(references=references[accent], predictions=predictions[accent])
        save_metrics(metrics, references, predictions, accent, wer)
        
    if verbose:
        print(f'\n DONE CALCULATING METRICS \n')

    return metrics


def main():
    pass


if __name__ == "__main__":
   main()