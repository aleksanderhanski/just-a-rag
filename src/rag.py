"""Answer questions over the indexed documents: retrieval + an LLM.

    uv run rag.py "your question here"

Retrieves the most relevant chunks from Qdrant, then asks Qwen (hosted free on
Groq) to answer using only those chunks, citing the source and page.
"""

import re
import sys

from dotenv import load_dotenv
from litellm import completion

from search import search

load_dotenv()   # loads GROQ_API_KEY from .env into the environment

LLM_MODEL = "groq/qwen/qwen3-32b"


def build_context(points):
    """Turn retrieved chunks into a numbered context block the model can cite."""
    blocks = []
    for i, point in enumerate(points, 1):
        p = point.payload
        blocks.append(f"[{i}] (source: {p['source']}, page {p['page']})\n{p['text']}")
    return "\n\n".join(blocks)


def answer(question, limit=5):
    points = search(question, limit=limit)
    if not points:
        return "No relevant documents found - have you run embed_data.py yet?"

    context = build_context(points)
    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the given context."
        "If the context doesn't contain the answer, say so plainly. Cite the "
        "sources you use by their bracket numbers, e.g. [1], [2]."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    text =response.choices[0].message.content
    # Drop Qwen's <think>...</think> reasoning
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text, points

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What are these documents about?"
    response, _ = answer(question)
    print(response)