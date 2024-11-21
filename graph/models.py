from typing import Literal
from pydantic import BaseModel, Field



class ObjectDetectionSchema(BaseModel):
    """Schema for object detection tool inputs."""

    vegetable: str = Field(..., description="Type of vegetable for detection.")
    characteristic: Literal["Ripened", "Fully Ripened", "All"] = Field(
        ...,
        description="Specifies the ripeness level or all characteristics for the vegetable."
    )
    threshold: float = Field(
        ..., description="Threshold to filter out low confidence detections."
    )