from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from app.core.logging import logger
from app.api.routes import router

app = FastAPI()
app.include_router(router)

logger.info("Starting app")

@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}