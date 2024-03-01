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

whisper_norm = BasicTextNormalizer()

def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch

def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": get_text(item), "norm_reference": item["norm_text"], "accents": get_accents(item)}


def main():
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    # device 0 for gpu, device -1 for cpu?
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=0
    ) # "openai/whisper-small"

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
    )

    whisper_asr.model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    print("processor/model loaded")

    # iterable dataset
    dataset_total = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", token=True, trust_remote_code=True, streaming=True)
    # dataset_total = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", token=True, trust_remote_code=True, streaming=True)
    text_column_name = "sentence"

    dataset_total = dataset_total.shuffle(seed=42, buffer_size=10_000)
    dataset_total = dataset_total.take(10_000) # 60k approx half of training # TODO: other 50k
    dataset_total = dataset_total.cast_column("audio", Audio(sampling_rate=16000))
    dataset_total = dataset_total.map(normalise) # , num_proc=2
    dataset_total = dataset_total.filter(is_target_text_in_range, input_columns=[text_column_name]) # , num_proc=2
    dataset_total = dataset_total.filter(lambda example: example['accent'] != '')

    print("dataset loaded")

    # make these specific per accent?
    predictions = {}
    references = {}
    norm_predictions = {}
    norm_references = {}

    all_accents = []

    i = 0

    for out in tqdm(whisper_asr(data(dataset_total), batch_size=16), desc='Decode Progress'):
        print(out)
        for accent in out["accents"][0]: # will skip if empty
            if accent not in all_accents:
                all_accents.append(accent)
            if accent not in predictions:
                predictions[accent] = []
            if accent not in references:
                references[accent] = []
            if accent not in norm_predictions:
                norm_predictions[accent] = []
            if accent not in norm_references:
                norm_references[accent] = []
            predictions[accent].append(out["text"])
            references[accent].append(out["reference"][0])
            norm_predictions[accent].append(whisper_norm(out["text"]))
            norm_references[accent].append(out["norm_reference"][0])

            i += 1
            if i % 100 == 0:
                print(i)
            with open('pred.pkl', 'wb') as f:
                pickle.dump(predictions, f)
            with open('norm_pred.pkl', 'wb') as f:
                pickle.dump(references, f)
            with open('ref.pkl', 'wb') as f:
                pickle.dump(norm_predictions, f)
            with open('norm_ref.pkl', 'wb') as f:
                pickle.dump(norm_references, f)

    metrics = {}

    for accent in all_accents:
        wer = wer_metric.compute(references=references[accent], predictions=predictions[accent])
        wer = round(100 * wer, 2)
        cer = cer_metric.compute(references=references[accent], predictions=predictions[accent])
        cer = round(100 * cer, 2)
        norm_wer = wer_metric.compute(references=norm_references[accent], predictions=norm_predictions[accent])
        norm_wer = round(100 * norm_wer, 2)
        norm_cer = cer_metric.compute(references=norm_references[accent], predictions=norm_predictions[accent])
        norm_cer = round(100 * norm_cer, 2)

        # print("\nACCENT: ", accent)
        # print("\nWER : ", wer)
        # print("CER : ", cer)
        # print("\nNORMALIZED WER : ", norm_wer)
        # print("NORMALIZED CER : ", norm_cer)

        acc_name = whisper_norm(accent).strip().replace(' ', '_')
        if acc_name == 'pakistan' or acc_name == 'sri_lanka':
            continue

        metrics[acc_name] = {'wer': wer, 'cer': cer, 'norm_wer': norm_wer, 'norm_cer': norm_cer}

        os.system(f"mkdir -p predictions")
        # os.system(f"mkdir -p predictions/{acc_name}")
        op_file = f"predictions/{acc_name}.txt" # ??
        result_file = open(op_file, 'w') # a? or w?
        result_file.write('ACCENT: ' + str(accent) + '\n')
        result_file.write('\nWER: ' + str(wer) + '\n')
        result_file.write('CER: ' + str(cer) + '\n')
        result_file.write('\nNORMALIZED WER: ' + str(norm_wer) + '\n')
        result_file.write('NORMALIZED CER: ' + str(norm_cer) + '\n\n\n')

        for ref, pred in zip(references[accent], predictions[accent]):
            result_file.write(f"REF: {ref.encode('utf-8')}\n")
            result_file.write(f"PRED: {pred.encode('utf-8')}\n")
            result_file.write("------------------------------------------------------" + '\n')
        result_file.close()

    print(metrics)


if __name__ == "__main__":
    print('starting')
    main()