from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import ml_pipeline_route


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.include_router(ml_pipeline_route.router)


@app.get("/")
def read_root():
    return {"Hello": "World"}
