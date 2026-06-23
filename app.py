"""Gradio chat UI for the RAG system.

    uv run app.py     # opens a local web UI at http://localhost:7860
"""

import gradio as gr

from rag import answer


def format_sources(points):
    lines = ["", "---", "**Sources**"]
    for i, point in enumerate(points, 1):
        p = point.payload
        lines.append(f"[{i}] {p['source']} - page {p['page']}")
    return "\n".join(lines)


def respond(message, history):
    """Answer one question. history is ignored for now - each query is independent."""
    response, points = answer(message)
    if points:
        response += "\n" + format_sources(points)
    return response


demo = gr.ChatInterface(
    fn=respond,
    title="Document Q&A",
    description="Ask questions about your indexed PDFs. Answers are grounded in the "
                "documents, with source citations.",
    examples=["What are these documents about?"],
)


if __name__ == "__main__":
    demo.launch()