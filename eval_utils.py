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

# modified from a github repo: https://github.com/vasistalodagala/whisper-finetune/tree/master
whisper_norm = BasicTextNormalizer()


def attach_peft(peft_model_id, commit=""):
    if commit:
        peft_config = PeftConfig.from_pretrained(peft_model_id, revision=commit)
    else:
        peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.config.use_cache = True

    return model

"""
Note: need to split metric by dialect for many-to-one.
assumes dataset is already filtered to only include intended accents, one per row
sd-qa: question, audio (source) // or question, accent, audio
cv: sentence, audio
"""
def new_evaluate(model, dataset, one_to_one=True):
    print("================================Beginning Evaluation================================")
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    metric = evaluate.load("wer")

    model_path = "openai/whisper-large-v2"
    task = "transcribe"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_path, task=task)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def prepare_features(data):
        # encode question text to label ids
        data["source_input_features"] = feature_extractor(np.asarray(data["audio"]["array"]), sampling_rate=data["audio"]["sampling_rate"]).input_features[0]
        
        if "sentence" in data: # CV
            data["accent"] = data["accents"]
            data["labels"] = tokenizer(data["sentence"]).input_ids
        elif "question" in data: # SD-QA
            data["labels"] = tokenizer(data["question"]).input_ids
        else:
            raise ValueError("Expected either CV or SD-QA dataset.")

        return data
    
    dataset = dataset.map(prepare_features, desc="Extract features")

    batch_size = 4
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="english", task="transcribe")
    normalizer = BasicTextNormalizer()

    if one_to_one:
        predictions, references, normalized_predictions, normalized_references = [], [], [], []
    else:
        predictions, references, normalized_predictions, normalized_references = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    model.eval()

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad(): # inference
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to(device), # check that this works
                        # forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

                if one_to_one:
                    predictions.extend(decoded_preds)
                    references.extend(decoded_labels)
                    normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                    normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
                else: # many to one
                    for i in range(batch_size):
                        accent = batch["accent"]
                        predictions[accent].append(decoded_preds[i])
                        references[accent].append(decoded_labels[i])
                        normalized_predictions[accent].append(normalizer(decoded_preds[i]).strip())
                        normalized_references[accent].extend(normalizer(decoded_labels[i]).strip())
            del generated_tokens, labels, batch
        gc.collect()

    if one_to_one:
        wer = 100 * metric.compute(predictions=predictions, references=references)
        normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
        eval_metrics = {"wer": wer, "normalized_wer": normalized_wer}
        total_wer = wer
    else:
        eval_metrics = {}
        total_wer = 0
        for accent in predictions:
            wer = 100 * metric.compute(predictions=predictions, references=references)
            normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
            eval_metrics[accent] = {"wer": wer, "normalized_wer": normalized_wer}
            total_wer += wer
        total_wer /= len(predictions)

    print(eval_metrics)
    print(f'(AVERAGED) WER: {total_wer}')
    return total_wer