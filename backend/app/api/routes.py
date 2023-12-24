from fastapi import APIRouter, HTTPException
from app.core.parser import parse_pdf
router = APIRouter()


@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/doc")
async def parse_document(path_to_doc: str):
    try:
        metadata = await parse_pdf(path_to_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"metadata": metadata}
