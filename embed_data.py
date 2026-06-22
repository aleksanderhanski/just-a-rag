"""Load PDFs from data/ and index them into Qdrant.

    python embed_data.py            # create collection if needed, then upsert all PDFs
    python embed_data.py --reset    # wipe and rebuild (after changing model/dimension/chunking)

Each PDF page is split into chunks small enough to stay under the embedding
model's token limit. Chunk ids are deterministic (uuid5 of "source:page:chunk"),
so re-running updates existing chunks in place and adds new PDFs - no duplicates.
"""

import sys
import uuid
from pathlib import Path

import pymupdf
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

from db import client, COLLECTION_NAME, MODEL_NAME, ensure_collection, reset_collection

DATA_DIR = Path("data")
CHUNK_SIZE = 800   # characters; ~150-200 tokens, comfortably under bge-base's 512-token limit


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks of up to chunk_size characters.

    Packs paragraphs (blank-line separated) together up to the limit so chunks
    respect semantic boundaries. A single paragraph longer than the limit is
    hard-split as a fallback, so no chunk ever exceeds chunk_size.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > chunk_size:
            if current:
                chunks.append(current); current = ""
            chunks.append(para[:chunk_size])
            para = para[chunk_size:]
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def load_chunks():
    """Return (text, source, page, chunk_index) for every chunk of every
    non-empty PDF page in data/."""
    records = []
    for path in DATA_DIR.glob("**/*.pdf"):
        with pymupdf.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text().strip()
                if not page_text:                     # skip blank / image-only pages
                    continue
                for chunk_index, chunk in enumerate(chunk_text(page_text)):
                    records.append((chunk, path.name, page_number, chunk_index))
    return records


def main():
    if "--reset" in sys.argv:
        reset_collection()
        print(f"Collection '{COLLECTION_NAME}' reset.")
    else:
        ensure_collection()

    records = load_chunks()
    if not records:
        raise SystemExit(f"No PDF pages with text found in {DATA_DIR}/")

    print(f"Embedding {len(records)} chunks with {MODEL_NAME} (first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [text for text, _, _, _ in records]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{page}:{chunk_index}")),
            vector=vec.tolist(),
            payload={"text": text, "source": source, "page": page, "chunk": chunk_index},
        )
        for (text, source, page, chunk_index), vec in zip(records, vectors)
    ]

    for i in range(0, len(points), 128):         # upsert in batches
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + 128])

    print(f"Upserted {len(points)} chunks into '{COLLECTION_NAME}'.")
    print("Total points in collection:", client.count(COLLECTION_NAME).count)


if __name__ == "__main__":
    main()