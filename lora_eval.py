# Import libraries
import os
from huggingface_hub import notebook_login
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration)
import torch
from peft import (PeftModel, 
                  PeftConfig)
from amy.old.eval_utils import (evaluate_asr,
                        get_mini_cv)

# os.system("pip install -q transformers librosa datasets==2.14.6 evaluate jiwer gradio bitsandbytes==0.37 accelerate geomloss gradio torchaudio")
# os.system("pip install -q git+https://github.com/huggingface/peft.git@main")

def main():
    # device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # log in to huggingface to save model as you go
    # notebook_login()

    # load whisper feature extractor, tokenizer, processor
    model_path = "openai/whisper-base"
    task = "transcribe"
    processor = WhisperProcessor.from_pretrained(model_path, task=task)

    print('LOADING MODEL')
    peft_model_id = "asyzhou/224n-whisper-base-alignment-milestone"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    print('DONE LOADING CONFIG')
    # TODO: verify that these are on the GPU
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, device_map="auto"
    ) # load_in_8bit=True # for quantized later?
    model = PeftModel.from_pretrained(model, peft_model_id) # attaches the PEFT module to the Whisper model
    model.config.use_cache = True

    pipe = model_pipeline(model, processor, verbose=True)

    print('GETTING DATASET')
    dataset = get_cv_split_mini() # .to(device)
    # TODO: FILTER THE DATASET["TRAIN"] FOR JUST THE LANGUAGES YOU WANT HERE

    print('EVALUATING')
    # metrics = evaluate_asr(model, processor, dataset["train"], True)
    # metrics = evaluate_asr_alt(pipe, dataset["train"], True)
    metrics = evaluate_asr_alt(pipe, dataset, True)
    print(metrics)

if __name__ == "__main__":
    # get_cv_split_mini()
    main()