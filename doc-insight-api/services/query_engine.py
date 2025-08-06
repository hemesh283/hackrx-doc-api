"""from openai import OpenAI
import os
import json
import re
import logging

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt(query: str, context: str) -> str:
    prompt = f"""
"""You are an intelligent assistant helping with insurance claims analysis.
Given the following document content and user query, extract the answer in structured JSON format.

--- Document Content ---
{context}
------------------------

User Question: {query}

IMPORTANT:
- Only return JSON, no markdown.
- Use this JSON format: {{ "answer": "..." }}
"""

"""    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    logging.info(f"Raw GPT Response: {content}")

    # Extract JSON safely
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try extracting JSON from text with regex fallback
        matches = re.findall(r"\{.*\}", content, re.DOTALL)
        for m in matches:
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue
    return {"answer": "Unable to parse response"}"""
