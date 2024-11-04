from pathlib import Path

BASE_DIR = Path(__file__).parent
DINO_MODULE = BASE_DIR / "tool_utils"
IMAGE_PATH = BASE_DIR / "tool_utils/tomatoes_dataset/garden-photos-4.jpg"
REFERENCE_IMAGE_PATH = BASE_DIR / "tool_utils/tomatoes_dataset"
ANNOTATED_IMAGE_PATH = DINO_MODULE / "tomatoes_dataset/annotated_image.jpg"
GENERATED_IMAGE_PATH = DINO_MODULE / "tomato_graph_outputs/annotated_graph_image.jpg"
ANNOTATED_REF_IMAGE_PATH = DINO_MODULE / "tomato_ref_annotated/annotated_reference_image.jpg"
