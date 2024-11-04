"""AI model util class definitions"""


class VisionModel:
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model
