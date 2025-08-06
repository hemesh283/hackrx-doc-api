from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import fitz  # PyMuPDF
import docx
import email
import email.policy
import os
import tempfile
import tiktoken
import json
import logging
from openai import OpenAI
from sqlalchemy.orm import Session
from db import SessionLocal
from models import Document, QueryLog
from uuid import UUID
import uuid

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DocInsight API",
    description="Upload and query documents (.pdf, .docx, .eml) using GPT-4o",
    version="1.0"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def extract_pdf_text(path: str) -> str:
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_docx_text(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_eml_text(path: str) -> str:
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_content())
    else:
        parts.append(msg.get_content())
    return "\n".join(parts)

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        suffix = os.path.splitext(file.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            text = extract_pdf_text(tmp_path)
        elif suffix == ".docx":
            text = extract_docx_text(tmp_path)
        elif suffix == ".eml":
            text = extract_eml_text(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        doc_id = str(uuid.uuid4())

        db_doc = Document(id=doc_id, filename=file.filename, filetype=suffix, content=text)
        db.add(db_doc)
        db.commit()

        return {"message": "File uploaded and parsed", "doc_id": doc_id}

    except Exception as e:
        logging.exception("File upload failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_query")
async def ask_query(query: str, doc_id: UUID, db: Session = Depends(get_db)):
    logging.info("=== ask_query endpoint STARTED ===")

    db_doc = db.query(Document).filter(Document.id == str(doc_id)).first()
    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found")

    full_text = db_doc.content

    # Truncate document to max token length
    max_tokens = 6000
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(full_text)
    truncated_text = encoding.decode(tokens[:max_tokens])

    prompt = (
        f"You are a helpful assistant specializing in insurance policy documents.\n\n"
        f"Your job is to analyze the following insurance policy text and answer the user's question in exactly one clear, concise sentence.\n"
        f"Only use the provided document text. If the document does not mention it, say so confidently, but do not say 'I don't know'.\n\n"
        f"--- Document Text ---\n"
        f"{truncated_text}\n"
        f"----------------------\n\n"
        f"User Question: {query}\n\n"
        f"Respond in this JSON format only:\n"
        f'{{"answer": "your single sentence answer here"}}'
    )

    try:
        logging.info("=== Sending prompt to OpenAI ===")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You answer strictly in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        logging.info(f"=== OpenAI Raw Response ===\n{content}")

        # Remove markdown formatting if present
        if content.startswith("```json"):
            content = content.strip("```json").strip("`").strip()
        elif content.startswith("```"):
            content = content.strip("```").strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid response from model: {content}")

        answer = parsed.get("answer", "No answer returned.")

        db.add(QueryLog(query_text=query, response_json=json.dumps(parsed)))
        db.commit()

        logging.info("=== ask_query endpoint COMPLETED ===")
        return {"answer": answer}

    except Exception as e:
        logging.exception("OpenAI query failed")
        raise HTTPException(status_code=500, detail="OpenAI query failed")

@app.get("/")
async def root():
    return {"message": "Welcome to DocInsight API 🚀"}

@app.get("/documents/")
def list_documents(db: Session = Depends(get_db)):
    try:
        docs = db.query(Document).all()
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "filetype": doc.filetype,
                "upload_time": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                "preview": doc.content[:100] + "..." if doc.content else ""
            }
            for doc in docs
        ]
    except Exception as e:
        logging.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail="Unable to fetch documents")

@app.delete("/delete/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        db.delete(doc)
        db.commit()
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        logging.exception("Failed to delete document")
        raise HTTPException(status_code=500, detail="Unable to delete document")
