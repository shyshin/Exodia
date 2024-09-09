from TTS.api import TTS


tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True).to("cuda")
tts.voice_conversion_to_file(source_wav="/home/yxpeng/Projects/RAGHack/Exodia/audio_output/output.wav", target_wav="/home/yxpeng/Projects/RAGHack/Exodia/music_sample/01.wav", file_path="/home/yxpeng/Projects/RAGHack/Exodia/audio_output/a2a01.wav")