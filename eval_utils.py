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

# modified from a github repo: https://github.com/vasistalodagala/whisper-finetune/tree/master
whisper_norm = BasicTextNormalizer()

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""
    
def get_text(sample):
    # can replace with just return sample["sentence"]?
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )
    
def get_accents(sample):
    if "accent" in sample:
        # can remove the india comma outlier thing
        return sample["accent"].split(',')
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
        yield {**item["audio"], "reference": get_text(item), "norm_reference": item["norm_text"], "accents": get_accents(item)}

def model_pipeline(model, processor, verbose=True):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor
    ) # , device=device
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


def get_half_cv():
    # iterable dataset
    dataset_total = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", token=True, streaming=True) # trust_remote_code=True, 
    text_column_name = "sentence"

    dataset_total = dataset_total.shuffle(seed=42, buffer_size=10_000)
    dataset_total = dataset_total.take(60_000) # 60k approx half of training
    dataset_total = dataset_total.cast_column("audio", Audio(sampling_rate=16000))
    dataset_total = dataset_total.map(normalise) # , num_proc=2
    dataset_total = dataset_total.filter(is_target_text_in_range, input_columns=[text_column_name]) # , num_proc=2
    dataset_total = dataset_total.filter(lambda example: example['accent'] != '')

    print("HALF CV DATASET LOADED")
    return dataset_total


def get_mini_cv():
    # iterable dataset
    dataset_total = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", token=True, streaming=True)
    text_column_name = "sentence"

    dataset_total = dataset_total.shuffle(seed=42, buffer_size=10_000)
    dataset_total = dataset_total.take(100) # 60k approx half of training
    dataset_total = dataset_total.cast_column("audio", Audio(sampling_rate=16000))
    dataset_total = dataset_total.map(normalise) # , num_proc=2
    dataset_total = dataset_total.filter(is_target_text_in_range, input_columns=[text_column_name]) # , num_proc=2
    # lol this is so jank
    dataset_total = dataset_total.filter(lambda example: 'Scottish English' in example['accent'] or 'India and South Asia (India' in example['accent'])

    print("MINI CV (ITERABLE) DATASET LOADED")
    return dataset_total


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

    for out in tqdm(asr_model(data(dataset_total), batch_size=16), desc='Decode Progress'):
        # print(out)
        for accent in out["accents"][0]: # will skip if empty
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
    if acc_name == 'pakistan' or acc_name == 'sri_lanka':
        return

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


def evaluate_asr(model, processor, dataset, verbose=True):
    wer_metric = evaluate.load("wer")

    whisper_asr = model_pipeline(model, processor, verbose)

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
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    print("WHISPER PROCESSOR/MODEL LOADED")

    # GET DATA
    print("LOADING DATA")
    # dataset = get_half_cv()
    dataset = get_mini_cv()

    metrics = evaluate_asr(model, processor, dataset, True)
    print(metrics)


if __name__ == "__main__":
    main()


# """
# EXAMPLE USAGE: (on common voice, iterable dataset ver)
# dataset = get_half_cv()
# metrics = evaluate_asr(model, processor, dataset)

# note: handles the accent column very specifically, so on SD-QA, either want to reformat to match with CV, or modify this code
# """
# from huggingface_hub import interpreter_login
# from datasets import load_dataset, Audio
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
# import evaluate
# from transformers.models.whisper.english_normalizer import BasicTextNormalizer
# from tqdm import tqdm
# import os
# import torch
# import pickle

# # modified from a github repo: https://github.com/vasistalodagala/whisper-finetune/tree/master
# whisper_norm = BasicTextNormalizer()

# def is_target_text_in_range(ref):
#     if ref.strip() == "ignore time segment in scoring":
#         return False
#     else:
#         return ref.strip() != ""
    
# def get_text(sample):
#     if "sentence" in sample: # COMMON VOICE
#         return sample["sentence"]
#     elif "question" in sample: # SD-QA
#         return sample["question"]
#     else:
#         raise ValueError(
#             f"Expected transcript column of either 'sentence' or 'question'. Got sample of "
#             ".join{sample.keys()}. Ensure a text column name is present in the dataset."
#         )

# # prob won't need this in the cleaned CV dataset bc the data's been exploded
# def get_accents(sample):
#     if "accent" in sample:
#         # can remove the india comma outlier thing
#         return sample["accent"].split(',')
#     else:
#         raise ValueError(
#             f"Expected transcript column of accent. Ensure an accent column is present in the dataset."
#         )

# def normalize(batch):
#     batch["norm_text"] = whisper_norm(get_text(batch))
#     return batch

# def data(dataset):
#     # MODIFY THIS FOR SD-QA SINCE GET_ACCENTS WON'T WORK
#     for i, item in enumerate(dataset):
#         if "accents" in item: # CV # TODO: switch this later
#             yield {**item["audio"], "reference": get_text(item), "norm_reference": item["norm_text"], "accents": get_accents(item)}
#         elif "question" in item: # SD-QA
#             yield {**item[source_dialect]["array"], "reference": get_text(item), "norm_reference": item["norm_text"], "accents": get_accents(item)}

# def model_pipeline(model, processor, verbose=True):
#     # device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
#     whisper_asr = pipeline(
#         "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor
#     ) # , device=device # discard device object when model uses the accelerate library
#     whisper_asr.model.config.forced_decoder_ids = (
#         whisper_asr.tokenizer.get_decoder_prompt_ids(
#             language="english", task="transcribe"
#         )
#     )
#     whisper_asr.model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
#         language="english", task="transcribe"
#     )
#     if verbose:
#         print("MODEL PIPELINE SET UP")
#     return whisper_asr


# def get_half_cv():
#     # iterable dataset
#     dataset_total = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", token=True, streaming=True) # trust_remote_code=True, 
#     text_column_name = "sentence"

#     dataset_total = dataset_total.shuffle(seed=42, buffer_size=10_000)
#     dataset_total = dataset_total.take(60_000) # 60k approx half of training
#     dataset_total = dataset_total.cast_column("audio", Audio(sampling_rate=16000))
#     dataset_total = dataset_total.map(normalize) # , num_proc=2
#     dataset_total = dataset_total.filter(is_target_text_in_range, input_columns=[text_column_name]) # , num_proc=2
#     dataset_total = dataset_total.filter(lambda example: example['accent'] != '')

#     print("HALF CV DATASET LOADED")
#     return dataset_total


# def get_mini_cv():
#     # iterable dataset
#     dataset_total = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", token=True, streaming=True)
#     text_column_name = "sentence"

#     dataset_total = dataset_total.shuffle(seed=42, buffer_size=10_000)
#     dataset_total = dataset_total.take(100) # 60k approx half of training
#     dataset_total = dataset_total.cast_column("audio", Audio(sampling_rate=16000))
#     dataset_total = dataset_total.map(normalize) # , num_proc=2
#     dataset_total = dataset_total.filter(is_target_text_in_range, input_columns=[text_column_name]) # , num_proc=2
#     # lol this is so jank
#     dataset_total = dataset_total.filter(lambda example: 'Scottish English' in example['accent'] or 'India and South Asia (India' in example['accent'])

#     print("MINI CV (ITERABLE) DATASET LOADED")
#     return dataset_total


# def pickle_dump(predictions, references, norm_predictions, norm_references):
#     os.system(f"mkdir -p pickles")
#     with open('pickles/pred.pkl', 'wb') as f:
#         pickle.dump(predictions, f)
#     with open('pickles/norm_pred.pkl', 'wb') as f:
#         pickle.dump(references, f)
#     with open('pickles/ref.pkl', 'wb') as f:
#         pickle.dump(norm_predictions, f)
#     with open('pickles/norm_ref.pkl', 'wb') as f:
#         pickle.dump(norm_references, f)


# def get_preds(asr_model, dataset_total, verbose):
#     predictions = {}
#     references = {}
#     norm_predictions = {}
#     norm_references = {}
#     all_accents = []

#     i = 0

#     for out in tqdm(asr_model(data(dataset_total), batch_size=16), desc='Decode Progress'):
#         # print(out)
#         for accent in out["accents"][0]: # will skip if empty
#             if accent not in all_accents:
#                 all_accents.append(accent)
#                 predictions[accent] = []
#                 references[accent] = []
#                 norm_predictions[accent] = []
#                 norm_references[accent] = []

#             predictions[accent].append(out["text"])
#             references[accent].append(out["reference"][0])
#             norm_predictions[accent].append(whisper_norm(out["text"]))
#             norm_references[accent].append(out["norm_reference"][0])

#             i += 1
#             if i % 100 == 0:
#                 if verbose:
#                     print(f'\niteration: {i}')
#                 pickle_dump(predictions, references, norm_predictions, norm_references)

#     print('all_accents: {all_accents}')
#     return predictions, references, norm_predictions, norm_references, all_accents


# def save_metrics(metrics, references, predictions, accent, wer, norm_wer):
#     acc_name = whisper_norm(accent).strip().replace(' ', '_')
#     if acc_name == 'pakistan' or acc_name == 'sri_lanka':
#         return

#     metrics[acc_name] = {'wer': wer, 'norm_wer': norm_wer}

#     os.system(f"mkdir -p evaluation")
#     op_file = f"evaluation/{acc_name}.txt"
#     result_file = open(op_file, 'w')
#     result_file.write('ACCENT: ' + str(accent) + '\n')
#     result_file.write('\nWER: ' + str(wer) + '\n')
#     result_file.write('\nNORMALIZED WER: ' + str(norm_wer) + '\n')

#     for ref, pred in zip(references[accent], predictions[accent]):
#         result_file.write(f"REF: {ref.encode('utf-8')}\n")
#         result_file.write(f"PRED: {pred.encode('utf-8')}\n")
#         result_file.write("------------------------------------------------------" + '\n')
#     result_file.close()


# def evaluate_asr(model, processor, dataset, verbose=True):
#     wer_metric = evaluate.load("wer")

#     whisper_asr = model_pipeline(model, processor, verbose)

#     # RUN INFERENCE
#     predictions, references, norm_predictions, norm_references, all_accents = get_preds(whisper_asr, dataset, verbose)
#     pickle_dump(predictions, references, norm_predictions, norm_references)

#     if verbose:
#         print('\n ASR COMPLETED\n')

#     # COMPUTE METRICS
#     wer_metric = evaluate.load("wer")
#     metrics = {}

#     for accent in all_accents:
#         wer = 100 * wer_metric.compute(references=references[accent], predictions=predictions[accent])
#         norm_wer = 100 * wer_metric.compute(references=norm_references[accent], predictions=norm_predictions[accent])
#         save_metrics(metrics, references, predictions, accent, wer, norm_wer)
        
#     if verbose:
#         print(f'\n DONE CALCULATING METRICS \n')

#     os.system(f"mkdir -p pickles")
#     with open('pickles/metrics.pkl', 'wb') as f:
#         pickle.dump(metrics, f)

#     return metrics


# def main():
#     # LOAD MODEL
#     model_name = "openai/whisper-base"
#     processor = WhisperProcessor.from_pretrained(model_name, language="English", task="transcribe")
#     model = WhisperForConditionalGeneration.from_pretrained(model_name)
#     print("WHISPER PROCESSOR/MODEL LOADED")

#     # GET DATA
#     print("LOADING DATA")
#     # dataset = get_half_cv()
#     dataset = get_mini_cv()

#     metrics = evaluate_asr(model, processor, dataset, True)
#     print(metrics)


# if __name__ == "__main__":
#     main()