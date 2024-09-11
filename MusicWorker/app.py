import gradio as gr
import os
import platform
from models.model import infer
import traceback


with gr.Blocks() as app:
    gr.HTML("<h1> RVC inference</h1>")

    audio_path = gr.Audio(
        label="Audio file",
        show_label=True,
        type="filepath",
    )

    with gr.Row():
        vc_output1 = gr.Textbox(label="Output")
        vc_output2 = gr.Audio(label="Output Audio")
    btn = gr.Button(value="Convert")
    btn.click(
        infer,
        inputs=[
            audio_path,
        ],
        outputs=[vc_output1, vc_output2],
        concurrency_limit=10,
    )

    app.launch(max_threads=200)
