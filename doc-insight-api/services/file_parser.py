import textract

def parse_document(filename: str, content: bytes) -> str:
    temp_path = f"temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    text = textract.process(temp_path).decode("utf-8")
    return text
