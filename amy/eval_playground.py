from huggingface_hub import interpreter_login
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, pipeline, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import pickle
import tqdm


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        batch["labels"] = [feature["labels"] for feature in features]

        return batch
    

def main():
    accelerator = Accelerator()
    # device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    dialects = [0] # switch to smth like "scottish_english" later

    metric = evaluate.load("wer")

    # LOAD MODEL
    model_name = "openai/whisper-base"
    task = "transcribe"
    language = "Hindi" # "English"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
        language="hi", task="transcribe"
    )
    print("WHISPER PROCESSOR/MODEL LOADED")

    # GET DATA
    print("PREPARING DATA")

    # prepare data
    cv = load_dataset("mozilla-foundation/common_voice_16_1", "hi", split="validation", token=True)
    # cv = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", token=True)
    
    # for testing purposes
    cv = cv.select(range(20))

    cv = cv.cast_column("audio", Audio(sampling_rate=16000))

    def keep_dialect(example):
        if not dialects: # no dialect filtering, evaluate all
            return True
        # for testing purposes # TODO: fix this
        return example['down_votes'] in dialects

    cv.filter(keep_dialect)

    def prepare_dataset(batch):
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # batch["labels"] = tokenizer(batch["sentence"]).input_ids
        batch["labels"] = batch["sentence"]
        return batch

    
    cv = cv.map(prepare_dataset) # , num_proc=2 # , remove_columns=cv.column_names["train"]

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    """
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    

    def compute_wer(pred, ref):
        return 100 * metric.compute(predictions=pred, references=ref)
    
    
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []

    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="amyguan/eval_test",
    #     do_train=False,
    #     evaluation_strategy="epoch",
    #     per_device_eval_batch_size=16
    # )

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     compute_metrics=compute_metrics,  # compute_metrics??? compute_wer??
    #     eval_dataset=cv
    # )
    """
    
    eval_dataloader = DataLoader(cv, batch_size=8, collate_fn=data_collator)

    # accelerator.prepare(eval_dataloader, model)

    all_predictions, all_references = [], []

    # for batch in eval_dataloader:
    for _, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad(): # run inference
            predicted_ids = model.generate(**batch)

        predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        all_predictions.extend(predictions)
        all_references.extend(batch["labels"])
        metric.add_batch(predictions=predictions, references=batch["labels"])

    wer = metric.compute()

    def save_items():
        os.system(f"mkdir -p pickles")
        with open('pickles/hi_pred.pkl', 'wb') as f:
            pickle.dump(all_predictions, f)
        with open('pickles/hi_ref.pkl', 'wb') as f:
            pickle.dump(all_references, f)

        os.system(f"mkdir -p evaluation")
        op_file = f"evaluation/hindi_test.txt"
        result_file = open(op_file, 'w')
        result_file.write('\nWER: ' + str(wer) + '\n')

        for ref, pred in zip(all_references, all_predictions):
            result_file.write(f"REF: {ref}\n")
            result_file.write(f"PRED: {pred}\n")
            result_file.write("------------------------------------------------------" + '\n')
        result_file.close()

    save_items()
    print(wer)


if __name__ == "__main__":
    main()