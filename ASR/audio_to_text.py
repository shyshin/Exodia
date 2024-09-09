import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import os
import numpy as np
from scipy.io import wavfile


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

## cuz cannot apt install ffmpeg, we have to do the fellowing convertion.
wav_file_path = '/home/yxpeng/Projects/RAGHack/Exodia/music_sample/01.wav'

sampling_rate, audio_data = wavfile.read(wav_file_path)

# Normalize the audio data to floating point (consistent with the array structure in the sample)
if audio_data.dtype == np.int16:
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize 16-bit PCM to the range [-1, 1]
elif audio_data.dtype == np.int32:
    audio_data = audio_data.astype(np.float32) / 2147483648.0  # Normalize 32-bit PCM to the range [-1, 1]

# Create a dictionary structure similar to the sample
sample_like_data = {
    'path': os.path.basename(wav_file_path),  # Get the filename as the path
    'array': np.array(audio_data[:, 0]),  # Single channel audio input
    'sampling_rate': sampling_rate  # Audio sampling rate
}

result = pipe(sample_like_data)
print(result["text"])

#result = pipe("/home/yxpeng/Projects/RAGHack/Exodia/music_sample/01.wav")
#print(result["text"])