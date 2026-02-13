from fastapi import FastAPI
from contextlib import asynccontextmanager

app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield
    print("Shutting down...")


@app.get("/")
def read_root():
    return {"Hello": "World"}
