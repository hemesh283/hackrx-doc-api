import uvicorn

if __name__ == "__main__":
    uvicorn.run("doc-insight-api.main:app", host="127.0.0.1", port=8080, reload=True)
