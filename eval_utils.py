"""
EXAMPLE USAGE: (on common voice, iterable dataset ver)
dataset = get_half_cv()
metrics = evaluate_asr(model, processor, dataset)

note: handles the accent column very specifically, so on SD-QA, either want to reformat to match with CV, or modify this code
"""
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

# modified from a github repo: https://github.com/vasistalodagala/whisper-finetune/tree/master
whisper_norm = BasicTextNormalizer()


def attach_peft(peft_model_id):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.config.use_cache = True

    return model

"""
Note: need to split metric by dialect for many-to-one.
"""
def new_evaluate(model, dataset):
    print("================================Beginning Evaluation================================")
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    metric = evaluate.load("wer")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding()

    model_path = "openai/whisper-large-v2"
    task = "transcribe"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_path, task=task)

    def prepare_features(data):
        # NOTE: UNSURE IF THE NP.ARRAY WORKS -- might need to troubleshoot
        data["source_input_features"] = feature_extractor(np.asarray(data["audio"]["array"]), sampling_rate=data["audio"]["sampling_rate"]).input_features[0]
        data["target_input_features"] = feature_extractor(np.asarray(data["audio"]["array"]), sampling_rate=data["audio"]["sampling_rate"]).input_features[0]
        
        # encode question text to label ids
        if "audio" in data: # CV (one to one)
            data["labels"] = tokenizer(data["audio"]["sentence"]).input_ids
        else: #  SD-QA
            # will need to pass in source(s), maybe keep a column for dialect so that i can separate 
            raise NotImplementedError("Have not implemented for SD-QA yet.")
        return data
    
    dataset.map(prepare_features, desc="Extract features")

    eval_dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    normalizer = BasicTextNormalizer()

    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad(): # inference
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to(device), # check that this works
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
            del generated_tokens, labels, batch
        gc.collect()

    wer = 100 * metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

    print(f"EVAL METRICS:\nWER: {wer}\nNORM_WER: {normalized_wer}")
    print(eval_metrics)



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