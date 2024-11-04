import sys
import logging
import io
import base64
from typing import Dict, Any
import numpy as np

from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

from .models import VisionModel
from .schemas import ImageMessageModel, TextMessageModel, AIMessage
from .settings import VISION_MODEL, PNG_FORMAT, LOGGER_FORMAT, SYS_IMG_ANALYSIS_PROMPT_TEST

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(LOGGER_FORMAT)
handler.setFormatter(formatter)

logger.addHandler(handler)

llm = ChatVertexAI(model=VISION_MODEL)
model = VisionModel(llm)


def handle_input(img: np.ndarray, prompt: str) -> str:
    """Handles the submission of input from the user.

    Args:
        img (np.ndarray): The image data in numpy array format.
        prompt (str): The prompt provided by the user.

    Returns:
        str: The AI-generated message based on the input image and prompt.
    """
    return generate_ai_response(img, prompt, model)


def _encode_input_img(image: np.ndarray) -> str:
    """Encodes an input image into a base64 string.

    Args:
        image (np.ndarray): The image to be encoded as a numpy array.

    Returns:
        str: Base64-encoded string representing the input image.
    """
    img = Image.fromarray(image)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=PNG_FORMAT)
    img_bytes.seek(0)
    img_bytes_content = img_bytes.getvalue()
    return base64.b64encode(img_bytes_content).decode("utf-8")


def generate_ai_response(
    img: np.ndarray, prompt: str, vision_model: VisionModel
) -> str:
    """Generates an AI response message based on the input image and prompt.

    Args:
        img (np.ndarray): The input image in numpy array format.
        prompt (str): The text prompt provided by the user.
        vision_model (VisionModel): The vision model used to generate the response.

    Returns:
        str: The AI-generated response message or an error message if the process fails.
    """
    logger.debug("Starting AI message generation process...")

    messages = create_ai_msg(img, prompt)
    ai_response = invoke_model(messages, vision_model)
    return extract_response_content(ai_response)


def create_ai_msg(img: np.ndarray, prompt: str) -> AIMessage:
    """Creates a message object by encoding the image and formatting the text prompt.

    Args:
        img (np.ndarray): The input image in numpy array format.
        prompt (str): The text prompt provided by the user.

    Returns:
        AIMessage: A namedtuple containing the formatted text and encoded image.
    """
    logger.debug(f"Encoding image and preparing message for prompt: {prompt}")
    encoded_img = _encode_input_img(img)

    image_msg = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"},
    }
    # Format the image analysis prompt with the user query.
    formatted_prompt = SYS_IMG_ANALYSIS_PROMPT_TEST.format(question=prompt)
    text_msg = {"type": "text", "text": f"{formatted_prompt}"}
    messages = {
        "text_msg": TextMessageModel(**text_msg),
        "image_msg": ImageMessageModel(**image_msg),
    }

    return AIMessage(**messages)


def invoke_model(messages: AIMessage, vision_model: VisionModel) -> Dict[str, Any]:
    """Invokes the AI model with the provided messages.

    Args:
        messages (AIMessage): The message object containing the prompt and encoded image.
        vision_model (VisionModel): The vision model used to generate the response.

    Returns:
        Dict[str, Any]: The AI model's response in dictionary format.

    Raises:
        Exception: If the model invocation fails.
    """
    logger.debug(
        f"Invoking the AI model with the given prompt: {messages.text_msg.text!r}."
    )

    msg = HumanMessage(content=[messages.text_msg.dict(), messages.image_msg.dict()])

    try:
        response = vision_model.model.invoke([msg]).dict()
        logger.debug("AI model invocation successful.")
        return response
    except Exception as e:
        logger.error(f"Error while invoking AI model: {str(e)}")
        raise


def extract_response_content(ai_response: Dict[str, Any]) -> str:
    """Extracts the content of the AI response.

    Args:
        ai_response (Dict[str, Any]): The response returned by the AI model.

    Returns:
        str: The extracted AI message content or an error message if the content is missing.
    """
    logger.debug("Extracting response content from AI model.")

    ai_msg = ai_response.get("content")
    if not ai_msg:
        logger.warning("AI model did not return a valid response.")
        return "Sorry, I don't have the answer. Please rephrase your question and try again."

    logger.debug(f"AI message: {ai_msg!r}")
    logger.debug(f"AI response metadata: {ai_response.get('usage_metadata')!r}")

    return ai_msg
