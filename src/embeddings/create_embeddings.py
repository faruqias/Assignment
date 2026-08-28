from pathlib import Path
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================
CHUNKS_FILE = Path(
    "data/extracted/attention/chunks.json"
)
OUTPUT_DIR = Path(
    "data/vectorstore/attention"
)
FAISS_INDEX_FILE = (
    OUTPUT_DIR / "index.faiss"
)
METADATA_FILE = (
    OUTPUT_DIR / "metadata.json"
)
MODEL_NAME = (
    "BAAI/bge-m3"
)
BATCH_SIZE = 8

NORMALIZE_EMBEDDINGS = True


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

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

    return chunks


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    chunk
):

    """
    Create the text that will be embedded.

    We intentionally include structural metadata because
    section/table/figure information is useful for semantic
    retrieval.
    """

    parts = []

    # --------------------------------------------------------
    # Content type
    # --------------------------------------------------------

    content_type = (
        chunk.get(
            "content_type"
        )
        or "text"
    )

    parts.append(
        f"Content Type: {content_type}"
    )

    # --------------------------------------------------------
    # Section
    # --------------------------------------------------------

    section_path = (
        chunk.get(
            "section_path"
        )
        or []
    )

    if section_path:

        parts.append(
            "Section: "
            + " > ".join(
                section_path
            )
        )

    # --------------------------------------------------------
    # Caption
    # --------------------------------------------------------

    caption = chunk.get(
        "caption"
    )

    if caption:

        parts.append(
            "Caption: "
            + caption
        )

    # --------------------------------------------------------
    # Main chunk text
    # --------------------------------------------------------

    text = chunk.get(
        "text",
        ""
    )

    if text:

        parts.append(
            text
        )

    return "\n\n".join(
        parts
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

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

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    model,
    chunks
):

    texts = [
        build_embedding_text(
            chunk
        )
        for chunk in chunks
    ]

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

        convert_to_numpy=True,

        normalize_embeddings=(
            NORMALIZE_EMBEDDINGS
        )
    )

    embeddings = (
        embeddings.astype(
            "float32"
        )
    )

    print()
    print(
        "Embedding generation complete."
    )

    print(
        f"Shape: {embeddings.shape}"
    )

    return embeddings


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(
    embeddings
):

    dimension = (
        embeddings.shape[1]
    )

    print()
    print(
        "Creating FAISS index..."
    )

    print(
        f"Vector dimension: {dimension}"
    )

    # --------------------------------------------------------
    # Because embeddings are normalized, Inner Product
    # behaves like cosine similarity.
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"Vectors indexed: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    chunks
):

    metadata = []

    for position, chunk in enumerate(
        chunks
    ):

        metadata.append(

            {
                "vector_id": position,

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

                "token_count": chunk.get(
                    "token_count"
                ),

                "parent_id": chunk.get(
                    "parent_id"
                ),

                "is_atomic": chunk.get(
                    "is_atomic",
                    False
                ),

                "caption": chunk.get(
                    "caption"
                ),

                "image_path": chunk.get(
                    "image_path"
                ),

                "document_part": chunk.get(
                    "document_part",
                    "main"
                ),

                "referenced_from": chunk.get(
                    "referenced_from",
                    []
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
        f"Metadata saved: "
        f"{METADATA_FILE}"
    )


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index
):

    faiss.write_index(
        index,
        str(
            FAISS_INDEX_FILE
        )
    )

    print(
        f"FAISS index saved: "
        f"{FAISS_INDEX_FILE}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_index(
    index,
    chunks
):

    print()
    print(
        "=" * 60
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Chunks              : {len(chunks)}"
    )

    print(
        f"FAISS vectors       : {index.ntotal}"
    )

    print(
        f"Vector dimension    : {index.d}"
    )

    print(
        f"Index metric        : Inner Product"
    )

    print(
        f"Normalized vectors  : {NORMALIZE_EMBEDDINGS}"
    )

    # --------------------------------------------------------
    # Validate counts
    # --------------------------------------------------------

    if index.ntotal != len(chunks):

        raise RuntimeError(
            "FAISS vector count does not "
            "match chunk count."
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    counts = {}

    for chunk in chunks:

        content_type = (
            chunk.get(
                "content_type",
                "unknown"
            )
        )

        counts[
            content_type
        ] = (
            counts.get(
                content_type,
                0
            )
            + 1
        )

    print()
    print(
        "Content distribution:"
    )

    for content_type, count in (
        counts.items()
    ):

        print(
            f"  {content_type:<10}: "
            f"{count}"
        )

    print()
    print(
        "Validation successful."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "STEP 4 - BGE-M3 + FAISS"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    if not chunks:

        raise RuntimeError(
            "No chunks found."
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        model,
        chunks
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = create_faiss_index(
        embeddings
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_index(
        index
    )

    save_metadata(
        chunks
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_index(
        index,
        chunks
    )

    print()
    print(
        "=" * 60
    )

    print(
        "STEP 4 COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()