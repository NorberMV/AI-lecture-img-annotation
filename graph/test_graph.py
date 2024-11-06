# This will call the desired graph
import sys
import json
from PIL import Image
from numpy import asarray
from typing import Dict, Any

# LangGraph related imports
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain related imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages

# Local imports
from .settings import (
    DINO_MODULE,
    CLEAN_IMAGE_PATH,
    GENERATED_IMAGE_PATH,
    REFERENCE_IMAGE_PATH,
    ANNOTATED_REF_IMAGE_PATH,
)
from .prompts import MAIN_ASSISTANT_SYS_PROMPT_TEMPLATE
from .agent import Agent
from .tools import object_detection_tool

sys.path.insert(0, DINO_MODULE.as_posix())
from .tool_utils.main import call_img_annotation
from .vision_model.utils import handle_input
from dotenv import load_dotenv
import pprint

# Load envvars
load_dotenv()

# Graph State Definition
class MyState(MessagesState):
    query: str
    image: str
    vegetable: str = ""
    characteristic: str
    threshold: float
    relatedness: bool
    image_analysis_json: str
    tool_call: str


agent = Agent(
    model_name="gpt-4o",
    temperature=0.0,
    tools=[object_detection_tool],
    system_prompt=MAIN_ASSISTANT_SYS_PROMPT_TEMPLATE,
)


# Node Definitions
def image_analysis_node(state: MyState) -> MyState:
    """
    This is the Image description node.
    """

    print("\n__ Node_1__: Running Image Analysis/Description!")
    img = Image.open(state["image"].as_posix())
    np_type_img = asarray(img)

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_query = msg.content

    print(
        f"Calling the Vision Model model with the following args: "
        f"{img!r}, {user_query!r}..."
    )
    img_analysis_json = handle_input(img=np_type_img, prompt=user_query)

    return {"image_analysis_json": img_analysis_json}


def tool_calling_llm(state: MyState) -> MyState:
    """
    Extraction values node
    """
    print("\n__ Node_2__: Running Tool Calling Node")

    extracted_values = parse_input(state["image_analysis_json"])
    print(
        f"\nThe parsed values from the Image Analysis Node are: "
        f"{extracted_values!r}"
    )
    relatedness = extracted_values["relatedness"]

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            query = [msg]

    if relatedness:
        # Format the tool call message
        tool_call_message = HumanMessage(
            content=(
                f"Given the following arguments:"
                f"\nvegetable: {extracted_values['vegetable_type']}"
                f"\ncharacteristic: {extracted_values['characteristic']}"
                f"\nthreshold: {extracted_values['threshold']}"
                f"\nPlease generate an annotated image according "
                f"to my question."
                f"\nQuestion:{query[0].content}"
            )
        )
        # Merges the two messages
        user_message = add_messages(agent.system_message, tool_call_message)
    else:
        user_message = add_messages(agent.system_message, query[0])
    print(f"\nThe full message send to the agent is: {user_message!r}")
    return {
        "messages": [agent.llm_with_tools.invoke(user_message)],
        "relatedness": relatedness,
        "vegetable": extracted_values["vegetable_type"],
        "characteristic": extracted_values["characteristic"],
        "threshold": extracted_values["threshold"],
        "query": query[0].content,
    }


def build_graph():
    """Constructs and compiles the state graph"""
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
    return graph


def run_graph(query: str):
    """
    Executes the image annotation graph.
    """
    graph = build_graph()
    messages = [HumanMessage(content=query)]
    answer = graph.invoke({"messages": messages, "image": ANNOTATED_REF_IMAGE_PATH})

    for ans in answer["messages"]:
        if isinstance(ans, AIMessage):
            answer["tool_call"] = ans.tool_calls
    _graph_prettier(answer)
    # This means the agent didn't use the tool,
    # so he previously replied.
    if not answer["relatedness"]:
        for ans in answer["messages"]:
            if isinstance(ans, AIMessage):
                return ans.content, None

    return (
        agent.llm_with_tools.invoke(
            [agent.system_message] + answer["messages"]
        ).content,
        GENERATED_IMAGE_PATH.as_posix(),
    )


def _graph_prettier(answer):
    print("\nA more pretty interaction messages version:\n")
    for m in answer["messages"]:
        m.pretty_print()
    print(
        "\n================================= Graph State ================================="
    )
    pprint.pprint(answer, indent=2)


# Some helper functions
def parse_input(img_analysis_json: str) -> Dict[str, Any]:
    """Parse JSON string and extract values"""
    print(f"The received image analysis str JSON is: {img_analysis_json!r}")
    if (" ```json\n" or " ```JSON\n") in img_analysis_json:
        img_analysis_json = img_analysis_json.strip(" ```json\n")
        img_analysis_json = img_analysis_json.strip(" ```JSON\n")

    parsed_data = json.loads(img_analysis_json)
    relatedness = parsed_data.get("relatedness")

    vegetable_type = parsed_data.get("vegetable_type", "")
    characteristic = parsed_data.get("characteristic", "")
    threshold_list = parsed_data.get("threshold", 0.5)
    if not relatedness:
        min_threshold = 0.0
    else:
        min_threshold = min(threshold_list)

    return {
        "vegetable_type": vegetable_type,
        "characteristic": characteristic,
        "relatedness": relatedness,
        "threshold": min_threshold,
    }
    # # we only care about relatedness
    # return {"relatedness": relatedness}


def annotate_reference(uploaded_img_path: str):
    TEXT_PROMPT = "ripened tomato"
    BOX_THRESHOLD = 0.26
    result = {
        "success": False,
        "message": "",
        "annotations": {"annotated_image_path": None, "threshold": BOX_THRESHOLD},
    }

    # Fixed Path
    reference_img_file = CLEAN_IMAGE_PATH
    print(f"The uploaded image object: {reference_img_file!r}")
    try:
        call_img_annotation(
            reference_img_file.as_posix(), TEXT_PROMPT, BOX_THRESHOLD, is_reference=True
        )
    except Exception as e:
        result["message"] = f"Error in image processing: {str(e)}"

    # Check if the annotated image exists at the expected path
    if REFERENCE_IMAGE_PATH.exists():
        result["success"] = True
        result["message"] = "Annotated reference image generated successfully."
        result["annotations"]["annotated_image_path"] = reference_img_file.as_posix()
    else:
        result[
            "message"
        ] = "Image annotation failed or image not found at expected path."

    print()
    print(result)
