from google.cloud import storage, firestore_v1
from app.logging import logger

try:
    db = firestore_v1.Client()
    storage = storage.Client()
    logger.info("Firebase initialized")
except Exception as e:
    logger.exception(e)
    db = None