# This will call the desired graph
import sys
from pathlib import Path
import json
import time
from datetime import datetime

import numpy as np
from PIL import Image
from numpy import asarray
from typing import List, Union, Literal, Optional, Dict, Any

# LangGraph related imports
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
# LangChain related imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.agents import tool
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field
from .settings import BASE_DIR, DINO_MODULE, IMAGE_PATH, ANNOTATED_IMAGE_PATH, GENERATED_IMAGE_PATH, \
    REFERENCE_IMAGE_PATH, ANNOTATED_REF_IMAGE_PATH
from .prompts import MAIN_ASSISTANT_SYS_PROMPT_TEMPLATE

sys.path.insert(0, DINO_MODULE.as_posix())
from .tool_utils import main
from .vision_node_utils.utils import handle_input
from dotenv import load_dotenv
import pprint

# Load envvars
load_dotenv()
GENERATE_MERMAID = False


class ObjectDetectionSchema(BaseModel):
    vegetable: str = Field(
        ..., description="The vegetable type the user is asking for"
    )
    characteristic: Literal["Ripened", "Fully Ripened", "All"] = Field(
        ...,
        description="The category of the characteristic required by the user into one of the following categories below according to the user question."
    )
    threshold: float = Field(
        ..., description="The threshold used to filter the detections."
    )
    relatedness: bool = Field(
        ..., description="True if the user query related to vegetables image detection, False otherwise."
    )


# Tool Definition
@tool(args_schema=ObjectDetectionSchema)
def object_detection_tool(vegetable: str, characteristic: str, threshold: float, relatedness: bool) -> dict:
    """
    Generate an annotated image based on the specified vegetable, characteristic, and detection threshold.

    Args:
        vegetable (str): The type of vegetable to be detected in the image.
        characteristic (str): The specific characteristic of the vegetable to focus on (e.g., "Ripened" or "Fully Ripened").
        threshold (float): Confidence threshold for object detection, filtering results below this confidence level.
        relatedness (bool): True if the user query related to vegetables image detection, False otherwise.
    """
    # Returns:
    #     dict: A JSON-formatted dictionary with detected objects and their annotations, including object type, confidence score, and positional data.
    formated_text_prompt = "{characteristic} {vegetable}".format(vegetable=vegetable, characteristic=characteristic)
    # Parameters for the request
    params = {
        # This will be the clean image, without annotations:
        # This is because the GroundDino model will be feed
        # with this clean image to be able to annotate it.
        "img_path": IMAGE_PATH,
        "text_prompt": formated_text_prompt,
        "threshold": threshold,
    }
    print(f"Calling the Object Detection Tool with the following args:\n{params!r}")

    result = {
        "success": False,
        "message": "",
        "annotations": {
            "annotated_image_path": None,
            "threshold": threshold
        },
        "original_image_info": {
            "path": IMAGE_PATH.as_posix(),
        }
    }

    try:
        main.call_vision_node(params["img_path"].as_posix(), params["text_prompt"], params["threshold"])

        # Check if the annotated image exists at the expected path
        if GENERATED_IMAGE_PATH.exists():
            result["success"] = True
            result["message"] = "Annotated image generated successfully."
            result["annotations"]["annotated_image_path"] = GENERATED_IMAGE_PATH.as_posix()
        else:
            result["message"] = "Image annotation failed or image not found at expected path."
    except Exception as e:
        result["message"] = f"Error in image processing: {str(e)}"

    return result


# Initializing Chat AI Model
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([object_detection_tool])

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initializing the system message, and other messages for the model.
chat_messages = [
    (
        "system",
        "You are a helpful image analysis bot assistant. Your goal is to detect if the user question imply he wants to annotate a given image In which case you should use the object detection tool.",
    ),
    ("human", "{query}.")  # ,
    # ("assistant", "Answer:")
]
chat_prompt = ChatPromptTemplate.from_messages(
    chat_messages
)

# System message
sys_msg = SystemMessage(content=MAIN_ASSISTANT_SYS_PROMPT_TEMPLATE)
# Bind the tools
# chat_with_tools = llm.bind_tools([object_detection_tool])
model_with_tools = llm.bind_tools([object_detection_tool])


# Graph State Definition
class MyState(MessagesState):
    query: str
    # image_path: str  # or AnyURL?
    image: np.array
    vegetable: str
    characteristic: str
    threshold: float
    relatedness: bool
    test_description: str  # A testing attribute, remove this
    tool_call: str


# First Node Definition
def image_analysis_node(state: MyState) -> MyState:
    """
    This is the Image description node.
    """
    # This node will look the annotated image.
    #  -> This node will return a message response with a JSON
    # String containing the necessary inputs for the tool call
    print("__ Node_1__: Running Image Analysis/Description!")
    print()
    print(f"This is the received state: {state!r}")
    # 1. Extract the annotated image path from the State, and the user query.
    # img_path = state["image_path"] # COMMENTED THIS
    # img = state["image"]
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
    # user_query = state.query
    # Process the annotated image to be a Numpy Array type.
    img = Image.open(state["image"].as_posix())  # COMMENTED THIS
    np_type_img = asarray(img)  # COMMENTED THIS
    # Then call the vision model with the retrieved arguments for analysis
    # and describe the annotated image.
    print(f"Calling the analysis image model with the following args: {img!r}, {user_query!r}...")
    img_analysis_json = handle_input(
        img=np_type_img,
        prompt=user_query
    )
    # This will return a JSON String answer, so I need to format that
    # answer and retrieve the values.
    # So here let's Update the rest of the State arguments.

    # maybe -> img_analysis_ans = json.loads(img_analysis_json)
    # vegetable = img_analysis_ans["vegetable"]
    # characteristic = img_analysis_ans["characteristic"]
    # threshold = img_analysis_ans["threshold"]
    # relatedness = ......

    # Then in the next node I should convert the previous
    # answer into an AIMessage.

    # TESTING RETURNED VALUE
    return {"test_description": img_analysis_json}
    # return {"vegetable":vegetable, ...}


# Second Node
# This is for the next node:
# return {"messages": [chat_with_tools.invoke(state["messages"])]}

# So in the next conditional edge (tool calling with tools_condition), I can simply extract the rest of the Graph State
# values and format the HumanMessage for doing the tool call This is for
# the edge transition: if relatedness = True:  messages: Can you annotate
# the image with the following data:
# {vegetable: val, characteristic: val, threshold: val}?
# Helper func
def parse_input(img_analysis_json: str) -> Dict[str, Any]:
    """Parse JSON string and extract values"""
    print(f"The received image analysis str JSON is: {img_analysis_json!r}")
    # Clean up and parse the JSON
    if (" ```json\n" or " ```JSON\n") in img_analysis_json:
        img_analysis_json = img_analysis_json.strip(" ```json\n")  # or img_analysis_json.strip("```JSON\n")
        img_analysis_json = img_analysis_json.strip(" ```JSON\n")  # or img_analysis_json.strip("```JSON\n")

    parsed_data = json.loads(img_analysis_json)
    print(f"THE PARSED DATA: {parsed_data!r}")
    relatedness = parsed_data.get("relatedness", True)  # default to True if not provided
    if relatedness:
        # Extract required values
        vegetable_type: str = parsed_data.get("vegetable_type", "")
        characteristic: str = parsed_data.get("characteristic", "")
        threshold_list: list = parsed_data.get("threshold", 0.5)  # default to 0.5 if not provided
        min_threshold = min(threshold_list)

        return {
            "vegetable_type": vegetable_type,
            "characteristic": characteristic,
            "relatedness": relatedness,
            "threshold": min_threshold
        }
    return {"relatedness": relatedness}


# Next edge definition:
# Node
def tool_calling_llm(state: MyState) -> MyState:
    """Extraction values node"""
    # Here I can simply extract the rest of the Graph State
    # # values and format the HumanMessage for doing the tool call This is for
    # # the edge transition: if relatedness = True:  messages: Can you annotate
    # # the image with the following data:
    # # {vegetable: val, characteristic: val, threshold: val}?
    # return {"messages": [llm_with_tools.invoke(state["messages"])]}
    print("Extracting values from the JSON and deciding if calling a tool or not.")
    # json.load()
    # relatedness = True
    print("Formatting the HumanMessage to contain the extracted values from the image analysis node.")

    # Call handle_input to parse img_analysis_json and extract values
    extracted_values = parse_input(state["test_description"])
    print()
    print(f"The extracted values are: {extracted_values!r}")
    relatedness = extracted_values["relatedness"]
    # FIXME: This should support more than a HumanMessage
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_message = [msg]
    # If relatedness it's True
    if relatedness:
        # Extract the values
        vegetable = extracted_values["vegetable_type"]
        characteristic = extracted_values["characteristic"]
        threshold = extracted_values["threshold"]
        # vegetable = "tomato"
        # characteristic = "ripened"
        # threshhold = 0.45
        # relatedness = False
        # Format the tool call message
        tool_call_message = HumanMessage(
            content=f"Given the following arguments:\nvegetable: {vegetable}\ncharacteristic: {characteristic}\nthreshold: {threshold}\nPlease generate an annotated image according to my question.\nQuestion:{user_message[0].content}")
        # Merges the two messages
        user_message = add_messages(sys_msg, tool_call_message)
        print(f"THE FULL MESSAGE SENDED TO THE TOOL IS: {user_message!r}")
    else:
        user_message = add_messages(sys_msg, user_message[0])
        # [llm_with_tools.invoke([sys_msg] + state["messages"])]
    # The question related to vegetable image annotation?
    # If True, format the user question with the prompt
    # TEST RETURN
    return {
        # "messages": [chat_with_tools.invoke([HumanMessage(content=tool_call_message)])],
        "messages": [model_with_tools.invoke(user_message)],
        "relatedness": relatedness
    }
    # return {"messages": [llm_with_tools.invoke(state["messages"])]}


# BUILD GRAPH
builder = StateGraph(MyState)
builder.add_node("image_analysis_node", image_analysis_node)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([object_detection_tool]))
builder.add_edge(START, "image_analysis_node")
builder.add_edge("image_analysis_node", "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()


def run_testing_graph(query: str, img_path: str = None):
    if not img_path:
        # Pass the annotated image as default
        img_path = ANNOTATED_IMAGE_PATH

    messages = [HumanMessage(content=query)]
    # Calling the Graph
    answer = graph.invoke({"messages": messages, "image_path": img_path})
    print("The resulting answer from the Graph is:\n\n")
    pprint.pprint(answer, indent=2)
    print()
    print("A more pretty interaction messages version:\n")
    for m in answer['messages']:
        m.pretty_print()

    # View
    if GENERATE_MERMAID:
        from IPython.display import Image, display
        graph_img = Image(
            graph.get_graph().draw_mermaid_png(output_file_path="mermaid_graph.png"))  # Directly passing the image data
        img = Image(filename="mermaid_graph.png")
        display(img)  # Display the image

    # # NOTE: Uncooment this just for testing each module used on the nodes isolated.
    # def run_graph():
    #     # Vars declaration
    #     img = Image.open(IMAGE_PATH)
    #     np_type_img = asarray(img)
    #     TEXT_PROMPT = "ripened tomato"
    #     BOX_THRESHOLD = 0.45
    #     TEXT_THRESHOLD = 0.01
    #
    #     # This is the DinoGrounding model
    #     main.call_vision_node(IMAGE_PATH.as_posix(), TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD)
    #     print("Calling vision module...")
    #     answer = handle_input(img=np_type_img, prompt="Can you describe the image?")
    #     print(answer)

    # That's how an invocation looks like:
    # messages = [HumanMessage(content="Hello world.")]
    # messages = graph.invoke({"messages": messages, "image_path": "/path/to/somewhere"})


def run_testing_graph2(img: np.array, query: str):
    # if not img.any():
    #     # Pass the annotated image as default
    #     img_path = ANNOTATED_IMAGE_PATH
    reference_img_file = ANNOTATED_REF_IMAGE_PATH
    print(f"The uploaded image object: {reference_img_file!r}")
    messages = [HumanMessage(content=query)]
    # Calling the Graph
    answer = graph.invoke({"messages": messages, "image": reference_img_file})
    print("The resulting answer from the Graph is:\n\n")
    pprint.pprint(answer, indent=2)
    print()
    print("A more pretty interaction messages version:\n")
    for m in answer['messages']:
        m.pretty_print()

    # View
    if GENERATE_MERMAID:
        from IPython.display import Image, display
        graph_img = Image(
            graph.get_graph().draw_mermaid_png(output_file_path="mermaid_graph.png"))  # Directly passing the image data
        img = Image(filename="mermaid_graph.png")
        display(img)  # Display the image
    generated_image = "graph/tool_utils/tomato_graph_outputs/annotated_graph_image.jpg"
    # This means the agent didn't used the tool, so he previously replied.
    if not answer["relatedness"]:
        for ans in answer["messages"]:
            if isinstance(ans, AIMessage):
                return ans.content, None #generated_image
    # I should be able to return also the Graph status as well
    return llm_with_tools.invoke([sys_msg] + answer["messages"]).content, generated_image


# if __name__ == "__main__":
#     description = image_analysis_node(MyState(messages=[HumanMessage(content=query)], image_path=ANNOTATED_IMAGE_PATH))
#     print(description)


def annotate_reference(uploaded_img_path: str):
    # # Vars declaration
    # img = Image.open(IMAGE_PATH)
    # np_type_img = asarray(img)
    TEXT_PROMPT = "ripened tomato"
    BOX_THRESHOLD = 0.26
    result = {
        "success": False,
        "message": "",
        "annotations": {
            "annotated_image_path": None,
            "threshold": BOX_THRESHOLD
        },
    }
    #
    # # This is the DinoGrounding model
    # main.call_vision_node(IMAGE_PATH.as_posix(), TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD)
    # print("Calling vision module...")
    # answer = handle_input(img=uploaded_img, prompt="Can you describe the image?")
    # print(answer)
    # Transform that into a Path obj
    # Dynamic Path
    # reference_img_name = Path(uploaded_img_path).name
    # reference_img_file = REFERENCE_IMAGE_PATH / reference_img_name
    # reference_img_file = IMAGE_PATH.as_posix()
    # Fixed Path
    reference_img_file = IMAGE_PATH
    print(f"The uploaded image object: {reference_img_file!r}")
    try:
        main.call_vision_node(reference_img_file.as_posix(), TEXT_PROMPT, BOX_THRESHOLD, is_reference=True)
    except Exception as e:
        result["message"] = f"Error in image processing: {str(e)}"

    # Check if the annotated image exists at the expected path
    if REFERENCE_IMAGE_PATH.exists():
        result["success"] = True
        result["message"] = "Annotated reference image generated successfully."
        result["annotations"]["annotated_image_path"] = reference_img_file.as_posix()
    else:
        result["message"] = "Image annotation failed or image not found at expected path."


    print()
    print(result)

# # NOTE: Uncooment this just for testing each module used on the nodes isolated.
# def run_graph():
#     # Vars declaration
#     img = Image.open(IMAGE_PATH)
#     np_type_img = asarray(img)
#     TEXT_PROMPT = "ripened tomato"
#     BOX_THRESHOLD = 0.45
#     TEXT_THRESHOLD = 0.01
#
#     # This is the DinoGrounding model
#     main.call_vision_node(IMAGE_PATH.as_posix(), TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD)
#     print("Calling vision module...")
#     answer = handle_input(img=np_type_img, prompt="Can you describe the image?")
#     print(answer)
#
# That's how an invocation looks like:
# messages = [HumanMessage(content="Hello world.")]
# messages = graph.invoke({"messages": messages, "image_path": "/path/to/somewhere"})
