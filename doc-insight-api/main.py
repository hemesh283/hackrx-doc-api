from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from some_module import process_document  # your own logic

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    result = process_document(content)
    return JSONResponse(content=result)
