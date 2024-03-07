import gradio as gr
import argparse
from model import SALMONN

class ff:
    def generate(self, wav_path, prompt, prompt_pattern, num_beams, temperature, top_p):
        print(f'wav_path: {wav_path}, prompt: {prompt}, temperature: {temperature}, num_beams: {num_beams}, top_p: {top_p}')
        return "I'm sorry, but I cannot answer that question as it is not clear what you are asking. Can you please provide more context or clarify your question?"

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--whisper_path", type=str, default=None)
parser.add_argument("--beats_path", type=str, default=None)
parser.add_argument("--vicuna_path", type=str, default=None)
parser.add_argument("--low_resource", action='store_true', default=False)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--port", default=9527)

args = parser.parse_args()

# model declaration
model = SALMONN(
    ckpt=args.ckpt_path,
    whisper_path=args.whisper_path,
    beats_path=args.beats_path,
    vicuna_path=args.vicuna_path,
    lora_alpha=args.lora_alpha,
    low_resource=args.low_resource
)
model.to(args.device)
model.eval()

# gradio 
def gradio_reset(chat_state):
    
    chat_state = []
    return (None,
            gr.update(value=None, interactive=True),
            gr.update(placeholder='Please upload your wav first', interactive=False),
            gr.update(value="Upload & Start Chat", interactive=True),
            chat_state)

def upload_speech(gr_speech, text_input, chat_state):
    
    if gr_speech is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state.append(gr_speech)
    return (gr.update(interactive=False),
            gr.update(interactive=True, placeholder='Type and press Enter'),
            gr.update(value="Start Chatting", interactive=False),
            chat_state)

def gradio_ask(user_message, chatbot, chat_state):
    
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state.append(user_message)
    chatbot.append([user_message, None])
    # 
    return gr.update(interactive=False, placeholder='Currently only single round conversations are supported.'), chatbot, chat_state

def gradio_answer(chatbot, chat_state, num_beams, temperature, top_p):
    llm_message = model.generate(
        wav_path=chat_state[0],
        prompt=chat_state[1],
        num_beams=num_beams,
        temperature=temperature,
        top_p=top_p,
    )
    chatbot[-1][1] = llm_message[0]
    return chatbot, chat_state

title = """<h1 align="center">SALMONN: Speech Audio Language Music Open Neural Network</h1>"""
image_src = """<h1 align="center"><a href="https://github.com/bytedance/SALMONN"><img src="https://raw.githubusercontent.com/bytedance/SALMONN/main/resource/salmon.png", alt="SALMONN" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>"""
description = """<h3>This is the demo of SALMONN. Upload your audio and start chatting!</h3>"""


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(image_src)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            speech = gr.Audio(label="Audio", type='filepath')
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True,
                label="top p",
            )

            temperature = gr.Slider(
                minimum=0.8,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=False,
                label="temperature",
            )

        with gr.Column():
            chat_state = gr.State([])
            
            chatbot = gr.Chatbot(label='SALMONN')
            text_input = gr.Textbox(label='User', placeholder='Please upload your speech first', interactive=False)
        
    upload_button.click(upload_speech, [speech, text_input, chat_state], [speech, text_input, upload_button, chat_state])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, num_beams, temperature, top_p], [chatbot, chat_state]
    )
    clear.click(gradio_reset, [chat_state], [chatbot, speech, text_input, upload_button, chat_state], queue=False)



demo.launch(share=True, enable_queue=True, server_port=int(args.port))