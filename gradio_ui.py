import gradio as gr

# # This will call the desired graph
try:
    from graph.test_graph import run_testing_graph2, annotate_reference
except ImportError as e:
    print("Error importing the local modules :(")
    raise e
else:
    print("COOL!, local graph module imported went smoothly!")


with gr.Blocks() as ui:
    with gr.Row():
        with gr.Column():
            # image_input_top = gr.Image(label="Upload image")
            image_input_top = gr.Image(label="Drop image here for processing", type="filepath")

            #image_input_bottom = gr.Image(label="Upload image")
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Your goal is to answer...")
            response = gr.Textbox(label="Answer", placeholder="Useful answer...")
            generated_image = gr.Image(label="Generated Image")  # Display the generated image


    submit_button = gr.Button("Submit")

    # Function to run when image is uploaded
    image_input_top.upload(
        fn=annotate_reference,
        inputs=image_input_top
        # outputs=response  # Assuming you want to output the result to the response box
    )


    submit_button.click(
        fn=run_testing_graph2,
        inputs=[image_input_top, prompt],
        outputs=[response, generated_image]
    )

ui.launch()
