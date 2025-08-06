import mimetypes
import docx
import PyPDF2
from openai import OpenAI

# Initialize OpenAI client (replace with your actual API key or load from env)
client = OpenAI(api_key="your-openai-api-key")

def parse_document(filename, content_bytes):
    mimetype, _ = mimetypes.guess_type(filename)

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(content_bytes)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(content_bytes)
    elif filename.endswith(".txt"):
        return content_bytes.decode("utf-8")
    else:
        raise ValueError("Unsupported file format: " + filename)

def extract_text_from_pdf(content_bytes):
    from io import BytesIO
    reader = PyPDF2.PdfReader(BytesIO(content_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(content_bytes):
    from io import BytesIO
    doc = docx.Document(BytesIO(content_bytes))
    return "\n".join([para.text for para in doc.paragraphs])


def ask_question(context, question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a helpful assistant that reads documents and answers questions in structured format."},
            {"role": "user", "content": f"Document:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()
