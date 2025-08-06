from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4
from services.file_parser import parse_document
from utils.store import document_store

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        parsed_text = parse_document(file.filename, contents)
        doc_id = str(uuid4())
        document_store[doc_id] = parsed_text
        return {"doc_id": doc_id, "message": "Document uploaded and parsed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
