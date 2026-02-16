import os
import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import ml_pipeline_route, predict_route
from utils.config import BEST_MODEL_PATH
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    if os.path.exists(BEST_MODEL_PATH):
        app.state.best_model = joblib.load(BEST_MODEL_PATH)
    else:
        logger.warning(
            "No best model found. Please train the model first and then predict."
        )
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.include_router(ml_pipeline_route.router)
app.include_router(predict_route.router)


@app.get("/")
def read_root():
    return {"Hello": "World"}
