import gradio as gr
import numpy as np

from utils import handle_input


with gr.Blocks() as ui:
    with gr.Row():
        with gr.Column():
            image_input_top = gr.Image(label="Upload image")
            image_input_bottom = gr.Image(label="Upload image")
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Your goal is to answer...")
            response = gr.Textbox(label="Answer", placeholder="Useful answer...")

    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=handle_input, inputs=[image_input_top, prompt], outputs=response
    )

ui.launch()
