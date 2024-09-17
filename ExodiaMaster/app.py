import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient
import os
import requests
import scipy.io.wavfile
import io
import time
from gradio_client import Client, file

# 获取 HuggingFace 的 API Token
client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.getenv('hf_token')
    #token=os.getenv('hf_token'),
)

def process_audio(audio_data):
    if audio_data is None:
        return "No audio provided.", ""

    # Check if audio_data is a tuple and extract data
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
    headers = {"Authorization": f"Bearer {os.getenv('hf_token')}"}

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

# Define a function to disable the button and display a loading indicator
def disable_components():
    # Update recognized_text content, indicating that processing is ongoing
    recognized_text_update = gr.update(value='Voice Recognition Running...')
    # Disable process_button
    process_button_update = gr.update(interactive=False)
    # Display loading animation
    loading_animation_update = gr.update(visible=True)
    return recognized_text_update, process_button_update, loading_animation_update

# Define a function to enable the button and hide the loading indicator
def enable_components(recognized_text):
    process_button_update = gr.update(interactive=True)
    # Hide loading animation
    loading_animation_update = gr.update(visible=False)
    return recognized_text, process_button_update, loading_animation_update

# Define a function to disable the button and display a loading indicator
def disable_chatbot_components():
    textbox = gr.update(interactive=False)
    submit_btn = gr.update(interactive=False)
    btn1 = gr.update(interactive=False)
    btn2 = gr.update(interactive=False)
    btn3 = gr.update(interactive=False)
    btn4 = gr.update(interactive=False)
    return textbox, submit_btn, btn1, btn2, btn3, btn4

# Define a function to enable the button and hide the loading indicator
def enable_chatbot_components():
    textbox = gr.update(interactive=True)
    submit_btn = gr.update(interactive=True)
    btn1 = gr.update(interactive=True)
    btn2 = gr.update(interactive=True)
    btn3 = gr.update(interactive=True)
    btn4 = gr.update(interactive=True)
    return textbox, submit_btn, btn1, btn2, btn3, btn4

llama_responded = 0
responded_answer = ""

def respond(
    message,
    history: list[tuple[str, str]]
):
    global llama_responded
    global responded_answer
    # Main Decision Module
    decision_response = ""
    judge_main_message = f"Here is a query: '{message}', Determine if this query is asking about one of the topics included in the list below. If it is, please directly provide only one name of the topic; Otherwise for any other queries, you just reply 'no'. The list of topics is: [movie, music, singing songs]"
    print(message)
    m_message = [{"role": "user", "content": judge_main_message}]
    for m in client.chat_completion(
        m_message,
        stream=True,
    ):
        token = m.choices[0].delta.content
        decision_response += token
    print(decision_response)

    if "movie" in decision_response.lower():
        movie_client = Client("ironserengety/movies-recommender")
        result = movie_client.predict(
                message=message,
                system_message="You are a movie recommender named 'Exodia'. You are extremely reliable. You always mention your name in the beginning of conversation. You will provide me with answers from the given info. Give not more than 3 choices and make sure that answers are complete sentences.",
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                api_name="/chat"
        )
        print(result)
        llama_responded = 1
        responded_answer = result
        return result
    
    elif "sing" in decision_response.lower() or "sing" in message.lower():
        llama_responded = 1
        responded_answer = "SING " + message
        return "Here is the song you might like!"
        
    else:
        #others
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
        print(messages)

        for message in client.chat_completion(
            messages,
            stream=True,
        ):
            token = message.choices[0].delta.content
            response += token

        llama_responded = 1
        responded_answer = response
        return response

def update_response_display():
    while not llama_responded:
        time.sleep(1)

def tts_part():
    global llama_responded
    global responded_answer
    result = ""
    if "SING" in responded_answer:
        client = Client("ironserengety/MusicRetriever")
        result = client.predict(
                message= responded_answer.lower(),
                api_name="/respond"
        )
        llama_responded = 0
        responded_answer = ""

    elif responded_answer != "" and responded_answer != "SING":
        text = responded_answer

        client = Client("tonyassi/voice-clone")
        result = client.predict(
                text,
                audio=file('siri.wav'),
                api_name="/predict"
        )
        llama_responded = 0
        responded_answer = ""
    return result

def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.HTML(
                value = '<h2 style="text-align: center;">Exodia AI Assistant</h2>'
            )
        # Audio input section
        with gr.Row():
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",  # Get audio data and sample rate
                label="Say Something..."
            )
            recognized_text = gr.Textbox(label="Recognized Text", interactive=False)
        
        # Process audio button
        process_button = gr.Button("Process Audio")
        
        # Loading animation
        loading_animation = gr.HTML(
            value='<div style="text-align: center;"><span style="font-size: 18px;">ASR Model is running...</span></div>',
            visible=False
        )

        # Chat interface using the custom chatbot instance
        chatbot = gr.ChatInterface(
            fill_height=True,
            fn=respond,
            submit_btn="Start Chatting"
        )
        user_start = chatbot.textbox.submit(
            fn=update_response_display, 
            inputs=[],
            outputs=[],
        )
        user_click = chatbot.submit_btn.click(
            fn=update_response_display, 
            inputs=[],
            outputs=[],
        )

        text_speaker = gr.Audio(
            label="Generated Audio"
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

        user_start.then(
            fn=disable_chatbot_components,
            inputs=[],
            outputs=[chatbot.submit_btn, chatbot.textbox, process_button, chatbot.retry_btn, chatbot.undo_btn, chatbot.clear_btn]
        ).then(
            fn=tts_part,
            inputs=[], 
            outputs=text_speaker
        ).then(
            fn=enable_chatbot_components,
            inputs=[],
            outputs=[chatbot.submit_btn, chatbot.textbox, process_button, chatbot.retry_btn, chatbot.undo_btn, chatbot.clear_btn]
        )

        user_click.then(
            fn=disable_chatbot_components,
            inputs=[],
            outputs=[chatbot.submit_btn, chatbot.textbox, process_button, chatbot.retry_btn, chatbot.undo_btn, chatbot.clear_btn]
        ).then(
            fn=tts_part,
            inputs=[], 
            outputs=text_speaker
        ).then(
            fn=enable_chatbot_components,
            inputs=[],
            outputs=[chatbot.submit_btn, chatbot.textbox, process_button, chatbot.retry_btn, chatbot.undo_btn, chatbot.clear_btn]
        )
                
    return demo



if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
