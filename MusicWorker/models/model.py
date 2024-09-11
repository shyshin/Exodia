from inference import Inference
import os


def infer(
    audio_file,
    model="SiriVT",
    f0_method="rmvpe",  # use rmvpe algorithm
    index_rate=0.75,  # Search feature ratio
    # 3 as ai agent is female
    vc_transform0=3,  # octave value: -12 for deep voices, 12 for female voices
    protect0=0.33,  # Protect voiceless consonants and breath sounds. 0.5 to disable it.
    resample_sr1=0,  # Re-sampling the output audio up to the final sample rate. 0 to not resample.
    filter_radius1=3,  # Filter (reduction of breathing harshness)
):
    if not model:
        return "No model url specified, please specify a model url.", None

    if not audio_file:
        return f"No audio file specified, please load an audio file: {audio_file}", None

    inference = Inference(
        model_name=model,
        f0_method=f0_method,
        source_audio_path=audio_file,
        feature_ratio=index_rate,
        transposition=vc_transform0,
        protection_amnt=protect0,
        resample=resample_sr1,
        harvest_median_filter=filter_radius1,
        output_file_name=os.path.join("./audio-outputs", os.path.basename(audio_file)),
    )

    output = inference.run()
    if "success" in output and output["success"]:
        print("Inference performed successfully...")
        return output, output["file"]
    else:
        print("Failure in inference...", output)
        return output, None
