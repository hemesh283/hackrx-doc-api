"""
from fastapi.responses import JSONResponse
from services.query_engine import ask_question
from utils.store import document_store

router = APIRouter()

@router.post("/")
async def query_doc(query: dict):
    try:
        user_query = query["query"]
        doc_id = query["doc_id"]

        if doc_id not in document_store:
            raise HTTPException(status_code=404, detail="Document ID not found")

        context = document_store[doc_id]
        answer = ask_question(context, user_query)

        return JSONResponse(content={
            "doc_id": doc_id,
            "query": user_query,
            "answer": answer
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) """
