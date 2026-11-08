from pydantic import BaseModel

from packs.text_classification.serving import ServingModel

WARMUP = "Stocks rally as the central bank holds rates"


class Request(BaseModel):
    text: str


def load(model_dir, settings) -> ServingModel:
    return ServingModel(model_dir, batch_size=settings.BATCH_SIZE)


def respond(model: ServingModel, req: Request) -> dict:
    return model.predict(req.text)
