import gradio as gr

# # This will call the desired graph
try:
    from graph.test_graph import run_graph, annotate_reference
except ImportError as e:
    print("Error importing the local modules :(")
    raise e
else:
    print("COOL!, local graph module imported went smoothly!")


with gr.Blocks() as ui:
    with gr.Row():
        with gr.Column():
            image_input_top = gr.Image(
                label="Drop image here for processing", type="filepath"
            )

        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Your goal is to answer...")
            response = gr.Textbox(label="Answer", placeholder="Useful answer...")
            generated_image = gr.Image(label="Generated Image")

    submit_button = gr.Button("Submit")

    # Function to run when image is uploaded
    image_input_top.upload(fn=annotate_reference, inputs=image_input_top)

    submit_button.click(
        fn=run_graph, inputs=[prompt], outputs=[response, generated_image]
    )

ui.launch()
