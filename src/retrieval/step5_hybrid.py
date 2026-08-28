import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
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

FAISS_INDEX_FILE = (
    VECTORSTORE_DIR / "index.faiss"
)

METADATA_FILE = (
    VECTORSTORE_DIR / "metadata.json"
)

RRF_OUTPUT_FILE = (
    VECTORSTORE_DIR / "rrf_candidates.json"
)

MODEL_NAME = "BAAI/bge-m3"

# Number of results from dense retrieval
DENSE_TOP_K = 10

# Number of results from BM25
BM25_TOP_K = 10

# Number of candidates passed to Step 6
RRF_TOP_K = 10

# RRF constant
RRF_K = 60


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 70)
print("STEP 5 - BM25 + RRF HYBRID RETRIEVAL")
print("=" * 70)

print()

print(
    f"Loading chunks: {CHUNKS_FILE}"
)

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(
    f"Loaded chunks: {len(chunks)}"
)


# ============================================================
# LOAD FAISS
# ============================================================

print()

print(
    "Loading FAISS index..."
)

index = faiss.read_index(
    str(FAISS_INDEX_FILE)
)

print(
    f"FAISS vectors: {index.ntotal}"
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


# ============================================================
# VALIDATION
# ============================================================

if len(chunks) != index.ntotal:

    raise RuntimeError(
        "Chunk count and FAISS vector count do not match."
    )


if len(metadata) != index.ntotal:

    raise RuntimeError(
        "Metadata count and FAISS vector count do not match."
    )


# ============================================================
# BUILD BM25 CORPUS
# ============================================================

print()

print(
    "Building BM25 index..."
)


def tokenize(text):

    return text.lower().split()


bm25_corpus = [

    tokenize(
        chunk.get(
            "text",
            ""
        )
    )

    for chunk in chunks

]


bm25 = BM25Okapi(
    bm25_corpus
)


print(
    f"BM25 documents: {len(bm25_corpus)}"
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
# DENSE SEARCH
# ============================================================

def dense_search(
    query,
    top_k=DENSE_TOP_K
):

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

    results = []

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

        results.append(
            {
                "vector_id": int(
                    vector_id
                ),

                "rank": rank,

                "score": float(
                    score
                )
            }
        )

    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query,
    top_k=BM25_TOP_K
):

    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_ids = np.argsort(
        scores
    )[::-1]

    results = []

    for rank, vector_id in enumerate(
        ranked_ids[:top_k],
        start=1
    ):

        results.append(
            {
                "vector_id": int(
                    vector_id
                ),

                "rank": rank,

                "score": float(
                    scores[vector_id]
                )
            }
        )

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=RRF_TOP_K,
    rrf_k=RRF_K
):

    fused_scores = {}

    dense_rank = {}

    bm25_rank = {}

    # --------------------------------------------------------
    # Dense ranking
    # --------------------------------------------------------

    for result in dense_results:

        vector_id = result[
            "vector_id"
        ]

        rank = result[
            "rank"
        ]

        dense_rank[
            vector_id
        ] = rank

        fused_scores[
            vector_id
        ] = (
            fused_scores.get(
                vector_id,
                0.0
            )
            +
            1.0 / (
                rrf_k + rank
            )
        )

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for result in bm25_results:

        vector_id = result[
            "vector_id"
        ]

        rank = result[
            "rank"
        ]

        bm25_rank[
            vector_id
        ] = rank

        fused_scores[
            vector_id
        ] = (
            fused_scores.get(
                vector_id,
                0.0
            )
            +
            1.0 / (
                rrf_k + rank
            )
        )

    # --------------------------------------------------------
    # Sort fused scores
    # --------------------------------------------------------

    ranked = sorted(

        fused_scores.items(),

        key=lambda x: x[1],

        reverse=True
    )

    results = []

    for vector_id, rrf_score in (
        ranked[:top_k]
    ):

        results.append(
            {
                "vector_id": int(
                    vector_id
                ),

                "rrf_score": float(
                    rrf_score
                ),

                "dense_rank": dense_rank.get(
                    vector_id
                ),

                "bm25_rank": bm25_rank.get(
                    vector_id
                )
            }
        )

    return results


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_result(
    rank,
    result
):

    vector_id = result[
        "vector_id"
    ]

    item = metadata[
        vector_id
    ]

    print()

    print(
        f"Rank: {rank}"
    )

    print(
        f"RRF Score: "
        f"{result['rrf_score']:.6f}"
    )

    print(
        f"Vector ID: {vector_id}"
    )

    print(
        f"Chunk ID: "
        f"{item.get('chunk_id')}"
    )

    print(
        f"Type: "
        f"{item.get('content_type')}"
    )

    print(
        f"Page: "
        f"{item.get('page_start')}-"
        f"{item.get('page_end')}"
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

    print(
        f"Dense Rank: "
        f"{result.get('dense_rank')}"
    )

    print(
        f"BM25 Rank: "
        f"{result.get('bm25_rank')}"
    )

    if item.get("caption"):

        print(
            "Caption:",
            item["caption"]
        )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query
):

    dense_results = dense_search(
        query
    )

    bm25_results = bm25_search(
        query
    )

    fused_results = reciprocal_rank_fusion(

        dense_results,

        bm25_results,

        top_k=RRF_TOP_K
    )

    # Attach query to every candidate
    for result in fused_results:

        result["query"] = query

    return (
        dense_results,
        bm25_results,
        fused_results
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


# ============================================================
# COLLECT ALL RRF RESULTS
# ============================================================

all_rrf_results = []


# ============================================================
# RUN TESTS
# ============================================================

for query in queries:

    print()

    print(
        "=" * 70
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 70
    )

    (
        dense_results,
        bm25_results,
        fused_results
    ) = hybrid_search(
        query
    )

    # --------------------------------------------------------
    # Store RRF results
    # --------------------------------------------------------

    all_rrf_results.extend(
        fused_results
    )

    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    print()

    print(
        "DENSE TOP RESULTS"
    )

    print(
        "-" * 70
    )

    for result in dense_results:

        item = metadata[
            result["vector_id"]
        ]

        print(
            f"{result['rank']:2}. "
            f"{result['score']:.4f} | "
            f"{item['chunk_id']} | "
            f"{item['content_type']}"
        )

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    print()

    print(
        "BM25 TOP RESULTS"
    )

    print(
        "-" * 70
    )

    for result in bm25_results:

        item = metadata[
            result["vector_id"]
        ]

        print(
            f"{result['rank']:2}. "
            f"{result['score']:.4f} | "
            f"{item['chunk_id']} | "
            f"{item['content_type']}"
        )

    # --------------------------------------------------------
    # RRF results
    # --------------------------------------------------------

    print()

    print(
        "RRF HYBRID TOP 10"
    )

    print(
        "-" * 70
    )

    for rank, result in enumerate(
        fused_results,
        start=1
    ):

        print_result(
            rank,
            result
        )


# ============================================================
# SAVE RRF CANDIDATES
# ============================================================

print()

print(
    "=" * 70
)

print(
    "SAVING RRF CANDIDATES"
)

print(
    "=" * 70
)


RRF_OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    RRF_OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_rrf_results,
        f,
        indent=2,
        ensure_ascii=False
    )


print()

print(
    f"RRF candidates saved:"
)

print(
    RRF_OUTPUT_FILE
)

print(
    f"Total candidates saved: "
    f"{len(all_rrf_results)}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()

print(
    "=" * 70
)

print(
    "STEP 5 VALIDATION"
)

print(
    "=" * 70
)

expected_candidates = (
    len(queries) * RRF_TOP_K
)

actual_candidates = (
    len(all_rrf_results)
)


print(
    f"Expected candidates : "
    f"{expected_candidates}"
)

print(
    f"Actual candidates   : "
    f"{actual_candidates}"
)


if actual_candidates != expected_candidates:

    raise RuntimeError(
        "RRF candidate count validation failed."
    )


# Validate query presence
missing_queries = [

    result

    for result in all_rrf_results

    if not result.get("query")
]


if missing_queries:

    raise RuntimeError(
        "Some RRF candidates are missing the query."
    )


# Validate vector IDs
invalid_vectors = [

    result

    for result in all_rrf_results

    if (
        result["vector_id"] < 0
        or
        result["vector_id"] >= len(chunks)
    )
]


if invalid_vectors:

    raise RuntimeError(
        "Invalid vector ID found in RRF candidates."
    )


print()

print(
    "Validation passed."
)

print(
    f"RRF Top K      : {RRF_TOP_K}"
)

print(
    f"Queries        : {len(queries)}"
)

print(
    f"Candidates     : {actual_candidates}"
)

print(
    f"Output         : {RRF_OUTPUT_FILE}"
)


# ============================================================
# FINAL
# ============================================================

print()

print(
    "=" * 70
)

print(
    "STEP 5 COMPLETED"
)

print(
    "=" * 70
)