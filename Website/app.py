from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from scipy.io import wavfile
import numpy as np
import sys

# 将项目的根目录 '/home/yxpeng/Projects/RAGHack/Exodia/' 添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
#print(sys.path)
from ASR.audio_to_text import asr
from Master_LLM.m_llm import inference
from Text2Speech.text_to_audio import tts_main

app = Flask(__name__)
socketio = SocketIO(app)

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 音频上传路由
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_data' not in request.files:
        return 'No audio file provided', 400
    
    # 获取上传的音频文件
    audio_file = request.files['audio_data']
    
    # 获取文件的原始文件名并强制设置为 .wav 扩展名
    original_filename = os.path.splitext(audio_file.filename)[0]  # 去掉原始文件的扩展名
    new_filename = f"{original_filename}.wav"  # 强制设置扩展名为 .wav
    
    # 保存文件到 uploads 目录
    save_path = os.path.join('uploads', new_filename)
    audio_file.save(save_path)

    # 使用 ASR 模型进行语音识别
    recognized_text = asr(save_path)
    print(recognized_text)
    
    # 通过 SocketIO 向前端发送识别结果
    socketio.emit('recognized_text', {'text': recognized_text})

    # 使用 Master LLM 进行推理
    master_response = inference(recognized_text)
    print(master_response)

    # TTS
    tts_main(master_response)
    
    # 通过 SocketIO 向前端发送音频播放请求
    socketio.emit('play_audio', {'audio_url': '/static/audio/TTS_output.wav'})

    return 'Audio file saved as .wav', 200

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    socketio.run(app, debug=True)
