from pydantic import BaseModel


class ImageMessageModel(BaseModel):
    type: str
    image_url: dict


class TextMessageModel(BaseModel):
    type: str
    text: str


class AIMessage(BaseModel):
    text_msg: TextMessageModel
    image_msg: ImageMessageModel
