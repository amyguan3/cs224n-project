"""
EXAMPLE USAGE: (on common voice, iterable dataset ver)
dataset = get_half_cv()
metrics = evaluate_asr(model, processor, dataset)

note: handles the accent column very specifically, so on SD-QA, either want to reformat to match with CV, or modify this code
"""
from huggingface_hub import interpreter_login
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm import tqdm
import os
import torch
import pickle
import numpy as np

# modified from a github repo: https://github.com/vasistalodagala/whisper-finetune/tree/master
whisper_norm = BasicTextNormalizer()

ACCENTS = ["Filipino",
           "United States English",
           "Scottish English",
           "Southern African (South Africa, Zimbabwe, Namibia)",
           "Hong Kong English",
           "India and South Asia (India, Pakistan, Sri Lanka)",
           "Australian English",
           "New Zealand English",
           "Irish English",
           "Kenyan English"
           ]

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""
    
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

def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch

def data(dataset):
    # MODIFY THIS FOR SD-QA SINCE GET_ACCENTS WON'T WORK
    for i, item in enumerate(dataset):
        yield {"raw": np.asarray(item["audio"]["array"]), "sampling_rate": item["audio"]["sampling_rate"], "reference": get_text(item), "norm_reference": item["norm_text"], "accents": get_accents(item)}

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

"""
returns cv with
cv["train"]
cv["test]
50% each
"""
def get_cv_split(accents=ACCENTS):
    cv_all = load_dataset("WillHeld/accented_common_voice", split="train", token=True, revision="e5b7f595177ccdb4a599f3589ce01957b0330357")
    cv_all = cv_all.shuffle(seed=42)
    cv_all = cv_all.select(range(10_000))

    # data split
    cv_split = cv_all.train_test_split(test_size=0.5, seed=42) # 28 samples in train
    cv_split = cv_split.map(normalise) # , num_proc=2
    cv_split = cv_split.filter(is_target_text_in_range, input_columns=["sentence"]) # , num_proc=2
    cv_split = cv_split.filter(lambda example: example['accents'] in accents)

    print("CV DATASET LOADED")

    return cv_split

def get_cv_split_mini(accents=ACCENTS):
    cv_all = load_dataset("WillHeld/accented_common_voice", split="train", token=True, revision="e5b7f595177ccdb4a599f3589ce01957b0330357")
    cv_all = cv_all.shuffle(seed=42)
    cv_all = cv_all.select(range(32))

    # data split
    cv_split = cv_all.train_test_split(test_size=0.5, seed=42) # 28 samples in train
    cv_split = cv_split.map(normalise) # , num_proc=2
    cv_split = cv_split.filter(is_target_text_in_range, input_columns=["sentence"]) # , num_proc=2
    cv_split = cv_split.filter(lambda example: example['accents'] in accents)

    print("MINI CV DATASET LOADED")
    return cv_split


def pickle_dump(predictions, references, norm_predictions, norm_references):
    os.system(f"mkdir -p pickles")
    with open('pickles/pred.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    with open('pickles/norm_pred.pkl', 'wb') as f:
        pickle.dump(references, f)
    with open('pickles/ref.pkl', 'wb') as f:
        pickle.dump(norm_predictions, f)
    with open('pickles/norm_ref.pkl', 'wb') as f:
        pickle.dump(norm_references, f)


def get_preds(asr_model, dataset_total, verbose):
    predictions = {}
    references = {}
    norm_predictions = {}
    norm_references = {}
    all_accents = []

    i = 0

    for out in tqdm(asr_model(data(dataset_total), batch_size=4), desc='Decode Progress'):
        # print(out)
        accent = out["accents"][0]
        if accent not in all_accents:
            all_accents.append(accent)
            predictions[accent] = []
            references[accent] = []
            norm_predictions[accent] = []
            norm_references[accent] = []

        predictions[accent].append(out["text"])
        references[accent].append(out["reference"][0])
        norm_predictions[accent].append(whisper_norm(out["text"]))
        norm_references[accent].append(out["norm_reference"][0])

        i += 1
        if i % 100 == 0:
            if verbose:
                print(f'\niteration: {i}')
            pickle_dump(predictions, references, norm_predictions, norm_references)

    return predictions, references, norm_predictions, norm_references, all_accents


def save_metrics(metrics, references, predictions, accent, wer, norm_wer):
    acc_name = whisper_norm(accent).strip().replace(' ', '_')

    metrics[acc_name] = {'wer': wer, 'norm_wer': norm_wer}

    os.system(f"mkdir -p evaluation")
    op_file = f"evaluation/{acc_name}.txt"
    result_file = open(op_file, 'w')
    result_file.write('ACCENT: ' + str(accent) + '\n')
    result_file.write('\nWER: ' + str(wer) + '\n')
    result_file.write('\nNORMALIZED WER: ' + str(norm_wer) + '\n')

    for ref, pred in zip(references[accent], predictions[accent]):
        result_file.write(f"REF: {ref.encode('utf-8')}\n")
        result_file.write(f"PRED: {pred.encode('utf-8')}\n")
        result_file.write("------------------------------------------------------" + '\n')
    result_file.close()

def evaluate_asr_alt(whisper_asr, dataset, verbose=True):
    wer_metric = evaluate.load("wer")

    # RUN INFERENCE
    predictions, references, norm_predictions, norm_references, all_accents = get_preds(whisper_asr, dataset, verbose)
    pickle_dump(predictions, references, norm_predictions, norm_references)

    if verbose:
        print('\n ASR COMPLETED\n')

    # COMPUTE METRICS
    wer_metric = evaluate.load("wer")
    metrics = {}

    for accent in all_accents:
        wer = 100 * wer_metric.compute(references=references[accent], predictions=predictions[accent])
        norm_wer = 100 * wer_metric.compute(references=norm_references[accent], predictions=norm_predictions[accent])
        save_metrics(metrics, references, predictions, accent, wer, norm_wer)
        
    if verbose:
        print(f'\n DONE CALCULATING METRICS \n')

    os.system(f"mkdir -p pickles")
    with open('pickles/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


def evaluate_asr(model, processor, dataset, verbose=True):
    wer_metric = evaluate.load("wer")

    whisper_asr = model_pipeline(model, processor, verbose, baseline=False)

    # RUN INFERENCE
    predictions, references, norm_predictions, norm_references, all_accents = get_preds(whisper_asr, dataset, verbose)
    pickle_dump(predictions, references, norm_predictions, norm_references)

    if verbose:
        print('\n ASR COMPLETED\n')

    # COMPUTE METRICS
    wer_metric = evaluate.load("wer")
    metrics = {}

    for accent in all_accents:
        wer = 100 * wer_metric.compute(references=references[accent], predictions=predictions[accent])
        norm_wer = 100 * wer_metric.compute(references=norm_references[accent], predictions=norm_predictions[accent])
        save_metrics(metrics, references, predictions, accent, wer, norm_wer)
        
    if verbose:
        print(f'\n DONE CALCULATING METRICS \n')

    os.system(f"mkdir -p pickles")
    with open('pickles/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


def main():
    # LOAD MODEL
    model_path = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(model_path, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    print("WHISPER PROCESSOR/MODEL LOADED")

    # GET DATA
    print("LOADING DATA")
    # dataset = get_cv_split()

    # metrics = evaluate_asr(model, processor, dataset["train"], True)

    dataset = get_cv_split_mini()

    metrics = evaluate_asr(model, processor, dataset["train"], True)
    print(metrics)


if __name__ == "__main__":
   """
EXAMPLE USAGE: (on common voice, iterable dataset ver)
dataset = get_half_cv()
metrics = evaluate_asr(model, processor, dataset)

note: handles the accent column very specifically, so on SD-QA, either want to reformat to match with CV, or modify this code
"""
from huggingface_hub import interpreter_login
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm import tqdm
import os
import torch
import pickle
import numpy as np

# modified from a github repo: https://github.com/vasistalodagala/whisper-finetune/tree/master
whisper_norm = BasicTextNormalizer()

ACCENTS = ["Filipino",
           "United States English",
           "Scottish English",
           "Southern African (South Africa, Zimbabwe, Namibia)",
           "Hong Kong English",
           "India and South Asia (India, Pakistan, Sri Lanka)",
           "Australian English",
           "New Zealand English",
           "Irish English",
           "Kenyan English"
           ]

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""
    
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

def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch

def data(dataset):
    # MODIFY THIS FOR SD-QA SINCE GET_ACCENTS WON'T WORK
    for i, item in enumerate(dataset):
        yield {"raw": np.asarray(item["audio"]["array"]), "sampling_rate": item["audio"]["sampling_rate"], "reference": get_text(item), "norm_reference": item["norm_text"], "accents": get_accents(item)}

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

"""
returns cv with
cv["train"]
cv["test]
50% each
"""

import pandas as pd

def get_cv_split(accents=ACCENTS):
    cv_all = load_dataset("WillHeld/accented_common_voice", split="train", token=True, revision="e5b7f595177ccdb4a599f3589ce01957b0330357")
    cv_all = cv_all.shuffle(seed=42)
    cv_all = cv_all.select(range(10_000)) # TODO: fix

    # data split
    cv_split = cv_all.train_test_split(test_size=0.5, seed=42) # 28 samples in train
    df = pd.DataFrame(cv_split["train"]["accents"])
    print(df.value_counts().head(20))
    # cv_split = cv_split.map(normalise) # , num_proc=2
    # cv_split = cv_split.filter(is_target_text_in_range, input_columns=["sentence"]) # , num_proc=2
    # cv_split = cv_split.filter(lambda example: example['accents'] in accents)

    print("CV DATASET LOADED")

    return cv_split

def get_cv_split_mini(accents=ACCENTS):
    cv_all = load_dataset("WillHeld/accented_common_voice", split="train", token=True, revision="e5b7f595177ccdb4a599f3589ce01957b0330357")
    cv_all = cv_all.shuffle(seed=42)
    cv_all = cv_all.select(range(32))

    # data split
    cv_split = cv_all.train_test_split(test_size=0.5, seed=42) # 28 samples in train
    cv_split = cv_split.map(normalise) # , num_proc=2
    cv_split = cv_split.filter(is_target_text_in_range, input_columns=["sentence"]) # , num_proc=2
    cv_split = cv_split.filter(lambda example: example['accents'] in accents)

    print("MINI CV DATASET LOADED")
    return cv_split


def pickle_dump(predictions, references, norm_predictions, norm_references):
    os.system(f"mkdir -p pickles")
    with open('pickles/pred.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    with open('pickles/norm_pred.pkl', 'wb') as f:
        pickle.dump(references, f)
    with open('pickles/ref.pkl', 'wb') as f:
        pickle.dump(norm_predictions, f)
    with open('pickles/norm_ref.pkl', 'wb') as f:
        pickle.dump(norm_references, f)


def get_preds(asr_model, dataset_total, verbose):
    predictions = {}
    references = {}
    norm_predictions = {}
    norm_references = {}
    all_accents = []

    i = 0

    for out in tqdm(asr_model(data(dataset_total), batch_size=4), desc='Decode Progress'):
        # print(out)
        accent = out["accents"][0]
        if accent not in all_accents:
            all_accents.append(accent)
            predictions[accent] = []
            references[accent] = []
            norm_predictions[accent] = []
            norm_references[accent] = []

        predictions[accent].append(out["text"])
        references[accent].append(out["reference"][0])
        norm_predictions[accent].append(whisper_norm(out["text"]))
        norm_references[accent].append(out["norm_reference"][0])

        i += 1
        if i % 100 == 0:
            if verbose:
                print(f'\niteration: {i}')
            pickle_dump(predictions, references, norm_predictions, norm_references)

    return predictions, references, norm_predictions, norm_references, all_accents


def save_metrics(metrics, references, predictions, accent, wer, norm_wer):
    acc_name = whisper_norm(accent).strip().replace(' ', '_')

    metrics[acc_name] = {'wer': wer, 'norm_wer': norm_wer}

    os.system(f"mkdir -p evaluation")
    op_file = f"evaluation/{acc_name}.txt"
    result_file = open(op_file, 'w')
    result_file.write('ACCENT: ' + str(accent) + '\n')
    result_file.write('\nWER: ' + str(wer) + '\n')
    result_file.write('\nNORMALIZED WER: ' + str(norm_wer) + '\n')

    for ref, pred in zip(references[accent], predictions[accent]):
        result_file.write(f"REF: {ref.encode('utf-8')}\n")
        result_file.write(f"PRED: {pred.encode('utf-8')}\n")
        result_file.write("------------------------------------------------------" + '\n')
    result_file.close()

def evaluate_asr_alt(whisper_asr, dataset, verbose=True):
    wer_metric = evaluate.load("wer")

    # RUN INFERENCE
    predictions, references, norm_predictions, norm_references, all_accents = get_preds(whisper_asr, dataset, verbose)
    pickle_dump(predictions, references, norm_predictions, norm_references)

    if verbose:
        print('\n ASR COMPLETED\n')

    # COMPUTE METRICS
    wer_metric = evaluate.load("wer")
    metrics = {}

    for accent in all_accents:
        wer = 100 * wer_metric.compute(references=references[accent], predictions=predictions[accent])
        norm_wer = 100 * wer_metric.compute(references=norm_references[accent], predictions=norm_predictions[accent])
        save_metrics(metrics, references, predictions, accent, wer, norm_wer)
        
    if verbose:
        print(f'\n DONE CALCULATING METRICS \n')

    os.system(f"mkdir -p pickles")
    with open('pickles/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


def evaluate_asr(model, processor, dataset, verbose=True):
    wer_metric = evaluate.load("wer")

    whisper_asr = model_pipeline(model, processor, verbose, baseline=False)

    # RUN INFERENCE
    predictions, references, norm_predictions, norm_references, all_accents = get_preds(whisper_asr, dataset, verbose)
    pickle_dump(predictions, references, norm_predictions, norm_references)

    if verbose:
        print('\n ASR COMPLETED\n')

    # COMPUTE METRICS
    wer_metric = evaluate.load("wer")
    metrics = {}

    for accent in all_accents:
        wer = 100 * wer_metric.compute(references=references[accent], predictions=predictions[accent])
        norm_wer = 100 * wer_metric.compute(references=norm_references[accent], predictions=norm_predictions[accent])
        save_metrics(metrics, references, predictions, accent, wer, norm_wer)
        
    if verbose:
        print(f'\n DONE CALCULATING METRICS \n')

    os.system(f"mkdir -p pickles")
    with open('pickles/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


def main():
    # # LOAD MODEL
    # model_path = "openai/whisper-large-v2"
    # processor = WhisperProcessor.from_pretrained(model_path, language="English", task="transcribe")
    # model = WhisperForConditionalGeneration.from_pretrained(model_path)
    # print("WHISPER PROCESSOR/MODEL LOADED")

    # GET DATA
    print("LOADING DATA")
    # dataset = get_cv_split()

    # metrics = evaluate_asr(model, processor, dataset["train"], True)

    dataset = get_cv_split()

    # metrics = evaluate_asr(model, processor, dataset["train"], True)
    # print(metrics)


if __name__ == "__main__":
   main()