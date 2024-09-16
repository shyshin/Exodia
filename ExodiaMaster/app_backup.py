import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient
import os
import requests
import scipy.io.wavfile
import io
import time
from langdetect import detect
from TTS.api import TTS

# 获取 HuggingFace 的 API Token
client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token="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"
    #token=os.getenv('hf_token'),
)

def process_audio(audio_data):
    if audio_data is None:
        return "No audio provided.", ""

    # 检查 audio_data 是否是元组，并提取数据
    if isinstance(audio_data, tuple):
        sample_rate, data = audio_data
    else:
        return "Invalid audio data format.", ""

    # Convert the audio data to WAV format in memory
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, data)
    wav_bytes = buf.getvalue()
    buf.close()

    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
    headers = {"Authorization": "Bearer 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"}

    def query(wav_data):
        response = requests.post(API_URL, headers=headers, data=wav_data)
        return response.json()

    # Call the API to process the audio
    output = query(wav_bytes)

    print(output) # Check output in console (logs in HF space)

    # Check the API response
    if 'text' in output:
        recognized_text = output['text']
        return recognized_text, recognized_text
    else:
        recognized_text = "The ASR module is still loading, please press the button again!"
        return recognized_text, ""

# 定义函数以禁用按钮并显示加载指示器
def disable_components():
    # 更新 recognized_text 的内容，提示用户正在处理
    recognized_text_update = gr.update(value='正在处理，请稍候...')
    # 禁用 process_button
    process_button_update = gr.update(interactive=False)
    # 显示加载动画
    loading_animation_update = gr.update(visible=True)
    return recognized_text_update, process_button_update, loading_animation_update

# 定义函数以启用按钮并隐藏加载指示器
def enable_components(recognized_text):
    process_button_update = gr.update(interactive=True)
    # 隐藏加载动画
    loading_animation_update = gr.update(visible=False)
    return recognized_text, process_button_update, loading_animation_update

llama_responded = 0
responded_answer = ""

def respond(
    message,
    history: list[tuple[str, str]]
):
    global llama_responded
    global responded_answer
    system_message = "You are a helpful chatbot that answers questions. Give any answer within 50 words."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        print(val[0])
        if val[0] != None:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        stream=True,
    ):
        token = message.choices[0].delta.content
        response += token

    llama_responded = 1
    responded_answer = response
    return response #gr.Audio("/home/yxpeng/Projects/RAGHack/Exodia/voice_sample/trump1.wav")

def update_response_display():
    while not llama_responded:
        time.sleep(1)

def bot(history):
    global llama_responded
    #print(history)
    history.append([None,gr.Audio("/home/yxpeng/Projects/RAGHack/Exodia/voice_sample/trump1.wav")])
    llama_responded = 0
   
    return history

def create_interface():
    with gr.Blocks() as demo:
        # Title
        gr.Markdown("# Exodia AI Assistant")
        
        # Audio input section
        with gr.Row():
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",  # Get audio data and sample rate
                label="Say Something..."
            )
            recognized_text = gr.Textbox(label="Recognized Text",interactive=False)
        
        # Process audio button
        process_button = gr.Button("Process Audio")
        
        # Loading animation
        loading_animation = gr.HTML(
            value='<div style="text-align: center;"><span style="font-size: 18px;">ASR Model is running...</span></div>',
            visible=False
        )

        # Chat interface using the custom chatbot instance
        chatbot = gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height=500),
            submit_btn="Start Chatting"
        )
        user_start =chatbot.textbox.submit(
            fn=update_response_display, 
            inputs=[],
            outputs=[],
        )

        # 在用户提交请求的时候
        #user_start = chatbot.textbox.submit()

        user_start.then(
            fn=bot,
            inputs=chatbot.chatbot, 
            outputs=chatbot.chatbot,  # 更新 response_display 的内容
        )

        # Associate audio processing function and update component states on click
        process_button.click(
            fn=disable_components,
            inputs=[],
            outputs=[recognized_text, process_button, loading_animation]
        ).then(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[recognized_text, chatbot.textbox]
        ).then(
            fn=enable_components,
            inputs=[recognized_text],
            outputs=[recognized_text, process_button, loading_animation]
        )

        # Layout includes Chatbot
        with gr.Row():
            chatbot_output = chatbot
                
    return demo



if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
