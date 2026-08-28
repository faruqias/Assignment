import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = Path(
    "data/extracted/attention/chunks.json"
)

VECTORSTORE_DIR = Path(
    "data/vectorstore/attention"
)

INDEX_FILE = (
    VECTORSTORE_DIR / "index.faiss"
)

METADATA_FILE = (
    VECTORSTORE_DIR / "metadata.json"
)

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 8

TOP_K = 5


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 60)
print("STEP 4 - BGE-M3 + FAISS")
print("=" * 60)

print()

print(
    f"Loading chunks from: {CHUNKS_FILE}"
)

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(
    f"Loaded {len(chunks)} chunks"
)


if not chunks:

    raise RuntimeError(
        "No chunks found."
    )


# ============================================================
# PREPARE TEXT
# ============================================================

texts = []

for chunk in chunks:

    text = chunk.get(
        "text",
        ""
    )

    if not text:

        text = (
            chunk.get(
                "caption",
                ""
            )
            or ""
        )

    texts.append(
        text.strip()
    )


print()

print(
    "Content distribution:"
)

content_counts = {}

for chunk in chunks:

    content_type = chunk.get(
        "content_type",
        "unknown"
    )

    content_counts[
        content_type
    ] = (
        content_counts.get(
            content_type,
            0
        )
        + 1
    )

for content_type, count in (
    content_counts.items()
):

    print(
        f"  {content_type:<10}: {count}"
    )


# ============================================================
# LOAD BGE-M3
# ============================================================

print()

print(
    "Loading BGE-M3..."
)

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "BGE-M3 loaded successfully."
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print()

print(
    "Creating embeddings..."
)

print(
    f"Chunks      : {len(texts)}"
)

print(
    f"Batch size  : {BATCH_SIZE}"
)

embeddings = model.encode(

    texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    normalize_embeddings=True,

    convert_to_numpy=True
)


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

print()

print(
    "Embedding generation complete."
)

print(
    f"Shape: {embeddings.shape}"
)

if embeddings.ndim != 2:

    raise RuntimeError(
        "Embeddings must be a 2D matrix."
    )


vector_dimension = (
    embeddings.shape[1]
)


print(
    f"Vector dimension: {vector_dimension}"
)


# ============================================================
# FAISS INDEX
# ============================================================

print()

print(
    "Creating FAISS index..."
)

# Inner Product + normalized vectors
# = cosine similarity

index = faiss.IndexFlatIP(
    vector_dimension
)

index.add(
    embeddings.astype(
        np.float32
    )
)

print(
    f"Vectors indexed: {index.ntotal}"
)


# ============================================================
# SAVE INDEX
# ============================================================

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

faiss.write_index(
    index,
    str(INDEX_FILE)
)

print()

print(
    f"FAISS index saved: {INDEX_FILE}"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = []

for vector_id, chunk in enumerate(
    chunks
):

    metadata.append(

        {
            "vector_id": vector_id,

            "chunk_id": chunk.get(
                "chunk_id"
            ),

            "document_id": chunk.get(
                "document_id"
            ),

            "document_name": chunk.get(
                "document_name"
            ),

            "content_type": chunk.get(
                "content_type"
            ),

            "page_start": chunk.get(
                "page_start"
            ),

            "page_end": chunk.get(
                "page_end"
            ),

            "section": chunk.get(
                "section"
            ),

            "section_path": chunk.get(
                "section_path",
                []
            ),

            "document_part": chunk.get(
                "document_part",
                "main"
            ),

            "caption": chunk.get(
                "caption"
            ),

            "text": chunk.get(
                "text",
                ""
            )
        }
    )


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Metadata saved: {METADATA_FILE}"
)


# ============================================================
# VALIDATION
# ============================================================

print()

print("=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    f"Chunks              : {len(chunks)}"
)

print(
    f"FAISS vectors       : {index.ntotal}"
)

print(
    f"Vector dimension    : {vector_dimension}"
)

print(
    "Index metric        : Inner Product"
)

print(
    "Normalized vectors  : True"
)

print()

print(
    "Content distribution:"
)

for content_type, count in (
    content_counts.items()
):

    print(
        f"  {content_type:<10}: {count}"
    )


# ============================================================
# FINAL CHECKS
# ============================================================

if index.ntotal != len(chunks):

    raise RuntimeError(
        "FAISS vector count does not match chunk count."
    )


if len(metadata) != index.ntotal:

    raise RuntimeError(
        "Metadata count does not match FAISS vector count."
    )


# ============================================================
# TEST SEARCH
# ============================================================

print()

print("=" * 60)
print("DENSE RETRIEVAL TEST")
print("=" * 60)


def search(query, top_k=5):

    query_embedding = model.encode(

        [query],

        normalize_embeddings=True,

        convert_to_numpy=True
    )

    scores, ids = index.search(

        query_embedding.astype(
            np.float32
        ),

        top_k
    )

    for rank, (
        score,
        vector_id
    ) in enumerate(
        zip(
            scores[0],
            ids[0]
        ),
        start=1
    ):

        if vector_id < 0:
            continue

        item = metadata[
            vector_id
        ]

        print()

        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            f"Vector ID: {vector_id}"
        )

        print(
            f"Chunk ID: {item['chunk_id']}"
        )

        print(
            f"Type: {item['content_type']}"
        )

        print(
            f"Page: "
            f"{item['page_start']}-"
            f"{item['page_end']}"
        )

        print(
            "Section:",
            " > ".join(
                item.get(
                    "section_path",
                    []
                )
            )
        )

        if item.get("caption"):

            print(
                "Caption:",
                item["caption"]
            )


# ============================================================
# TEST QUERIES
# ============================================================

queries = [

    "How does scaled dot-product attention work?",

    "What are the different variations of the Transformer architecture?",

    "Which attention heads appear to perform anaphora resolution?",

    "What is the computational complexity of self-attention?"

]


for query in queries:

    print()

    print(
        "=" * 60
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 60
    )

    search(
        query,
        TOP_K
    )


print()

print("=" * 60)

print(
    "STEP 4 COMPLETED"
)

print("=" * 60)