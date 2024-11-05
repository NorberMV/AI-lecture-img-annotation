import sys

from langchain.agents import tool

from .settings import DINO_MODULE
sys.path.insert(0, DINO_MODULE.as_posix())
from .tool_utils.main import call_img_annotation
from .models import ObjectDetectionSchema
from .settings import CLEAN_IMAGE_PATH, GENERATED_IMAGE_PATH


# Tool Definition
@tool(args_schema=ObjectDetectionSchema)
def object_detection_tool(
    vegetable: str, characteristic: str, threshold: float
) -> dict:
    """Annotates an image to highlight specified vegetable characteristics.

    Args:
        vegetable (str): The type of vegetable to detect in the image.
        characteristic (str): The specific characteristic to focus on (e.g., "Ripened").
        threshold (float): The confidence threshold for object detection.

    Returns:
        dict: Metadata about the annotation process including success status,
              message, and path to the annotated image if successful.
    """
    formatted_text_prompt = "{characteristic} {vegetable}".format(
        vegetable=vegetable, characteristic=characteristic
    )
    vision_params = {
        # This will be the clean image, without annotations:
        # This is because the GroundDino model will be feed
        # with this clean image to be able to annotate it.
        "img_path": CLEAN_IMAGE_PATH,
        "text_prompt": formatted_text_prompt,
        "threshold": threshold,
    }
    annotation_metadata = {
        "success": False,
        "message": "",
        "annotations": {"annotated_image_path": None, "threshold": threshold},
        "original_image_info": {
            "path": CLEAN_IMAGE_PATH.as_posix(),
        },
    }
    print(
        f"Calling the Object Detection Tool with the following args:\n{vision_params!r}"
    )
    try:
        call_img_annotation(
            vision_params["img_path"].as_posix(),
            vision_params["text_prompt"],
            vision_params["threshold"],
        )
        # Check if the annotated image exists at the expected path
        if GENERATED_IMAGE_PATH.exists():
            annotation_metadata["success"] = True
            annotation_metadata["message"] = "Annotated image generated successfully."
            annotation_metadata["annotations"][
                "annotated_image_path"
            ] = GENERATED_IMAGE_PATH.as_posix()
        else:
            annotation_metadata[
                "message"
            ] = "Image annotation failed or image not found at expected path."
    except Exception as e:
        annotation_metadata["message"] = f"Error in image processing: {str(e)}"
        print(f"There was an error annotating the image:\nstr(e)")

    return annotation_metadata