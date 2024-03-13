import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from qa_metrics.cfm import CFMatcher
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from transformers.generation import GenerationConfig

from salmonn import SALMONN

torch.manual_seed(1234)
cfm = CFMatcher()

dials = [
    "aus",
    "gbr",
    "ind_n",
    "ind_s",
    "irl",
    "kenya",
    "nga",
    "nzl",
    "phl",
    "usa",
    "zaf",
]


@torch.no_grad()
def get_response_pipeline(asr_model, model, audio, dial):
    value = audio[dial]
    text = asr_model(value)["text"]
    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Give answers as a simple single sentence.",
        },
        {"role": "user", "content": text},
    ]
    # Mistral
    # chat = [{"role": "user", "content": text + " Give a simple one sentence answer."}]
    query = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")

    output = model.generate(query, max_new_tokens=128)
    # Mistral & Llama
    if "qwen" not in model.name:
        split_token = "[/INST]"
        response = (
            tokenizer.decode(output[0], skip_special_tokens=True)
            .split(split_token)[-1]
            .strip()
        )
    else:
        split_token = "<|im_start|>"
        response = (
            tokenizer.decode(output[0], skip_special_tokens=False)
            .split(split_token)[-1]
            .strip()
            .replace("<|im_end|>", "")
        )

    scores = [
        value[response]
        for value in cfm.get_scores(
            audio["answers"], response, audio["question"]
        ).values()
    ]
    return response, max(scores)


@torch.no_grad()
def get_response_pipeline_qwen(asr_model, model, audio, dial):
    value = audio[dial]
    text = asr_model(value)["text"]

    query = tokenizer.from_list_format(
        [
            {"text": text + " Give a simple one sentence answer."},
        ]
    )

    response, history = model.chat(tokenizer, query=query, history=None)
    scores = [
        value[response]
        for value in cfm.get_scores(
            audio["answers"], response, audio["question"]
        ).values()
    ]
    return response, max(scores)


@torch.no_grad()
def get_response_end_to_end_s(model, audio, dial):
    value = audio[dial]
    sf.write("tmp.wav", value["array"], value["sampling_rate"], format="wav")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = model.generate(
            wav_path="tmp.wav",
            prompt="You are a helpful assistant. Give a simple one sentence answer.",
            do_sample=False,
            top_p=1.0,
        )
    response = llm_message[0]

    scores = [
        value[response]
        for value in cfm.get_scores(
            audio["answers"], response, audio["question"]
        ).values()
    ]
    return response, max(scores)


@torch.no_grad()
def get_response_end_to_end_q(model, audio, dial):
    value = audio[dial]
    sf.write("tmp.wav", value["array"], value["sampling_rate"], format="wav")
    query = tokenizer.from_list_format(
        [{"audio": "tmp.wav"}, {"text": "Give a simple one sentence answer."}]
    )

    response, history = model.chat(
        tokenizer,
        query=query,
        system="You are a helpful assistant.",
        history=None,
    )
    scores = [
        value[response]
        for value in cfm.get_scores(
            audio["answers"], response, audio["question"]
        ).values()
    ]
    return response, max(scores)


m_type = "e2e"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "Qwen/Qwen-Audio-Chat"
# model_name = "Qwen/Qwen1.5-7B-Chat"
model_name = "salmonn_7b"
if m_type == "e2e":
    if "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).eval()

        model.generation_config = GenerationConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            do_sample=False,
            top_k=50,
            top_p=1.0,
        )
    elif "salmonn" in model_name:
        model = SALMONN(
            ckpt="./SALMONN_PATHS/salmonn_7b_v0.pth",
            # whisper_path="./SALMONN_PATHS/whisper-large-v2",
            whisper_path="openai/whisper-large-v2",  # NEW
            beats_path="./SALMONN_PATHS/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            # vicuna_path="./SALMONN_PATHS/vicuna-7b-v1.5",
            vicuna_path="lmsys/vicuna-13b-v1.1",  # NEW
            low_resource=False,
        )

else:
    asr_model_id = "openai/whisper-large-v3"

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_model_id,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    processor = AutoProcessor.from_pretrained(asr_model_id)

    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        do_sample=False,
        max_new_tokens=1000,
        top_k=50,
        top_p=1.0,
        temperature=1.0,
    )

ds = load_dataset("WillHeld/SD-QA")["dev"].filter(lambda example: example["answers"])
dial_scores = {}
for dial in dials:
    scores = []
    with open(f"./sdqa-res/salmonn_7b/{dial}_outs.txt", "w") as f:
        for idx, ex in enumerate(tqdm(ds)):
            try:
                id = ex["id"]
                if m_type == "e2e" and "qwen" in model_name.lower():
                    pred, score = get_response_end_to_end_q(model, ex, dial)
                elif m_type == "e2e" and "salmonn" in model_name.lower():
                    pred, score = get_response_end_to_end_s(model, ex, dial)
                elif m_type != "e2e" and (
                    "qwen" in model_name.lower() and "1.5" not in model_name
                ):
                    pred, score = get_response_pipeline_qwen(asr, model, ex, dial)
                else:
                    pred, score = get_response_pipeline(asr, model, ex, dial)
            except Exception as e:
                print(e)
                print(id)
                pred = "ERROR IN PROCESSING"
                score = 0.0
            scores.append(score)
            pred_rep = '"""' + pred.replace("\n", "[NEW_LINE]") + '"""'
            f.write(f"{id}[SEP_DIAL]{pred_rep}[SEP_DIAL]{score}\n")
            if idx % 10 == 0 and idx != 0:
                f.flush()
    dial_scores[dial] = np.mean(scores)
print(dial_scores)