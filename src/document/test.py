import json

from retriever.reranker import BGEReranker


CHUNKS_FILE = (
    "data/extracted/attention/chunks.json"
)

RRF_FILE = (
    "data/vectorstore/attention/rrf_candidates.json"
)


# ============================================================
# LOAD
# ============================================================

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


with open(
    RRF_FILE,
    "r",
    encoding="utf-8"
) as f:

    candidates = json.load(f)


# ============================================================
# RERANKER
# ============================================================

reranker = BGEReranker()


# ============================================================
# TEST
# ============================================================

query = (
    "How does scaled dot-product attention work?"
)

# Your rrf_candidates.json contains multiple queries,
# so filter this query.

query_candidates = [
    candidate
    for candidate in candidates
    if candidate.get("query") == query
]


# ============================================================
# RUN
# ============================================================

results = reranker.rerank(
    query,
    query_candidates,
    chunks
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 60)
print("RERANKER TEST")
print("=" * 60)

print(
    f"Enabled: {reranker.enabled}"
)

print(
    f"Results: {len(results)}"
)


for rank, result in enumerate(
    results,
    start=1
):

    print()

    print(
        f"Rank       : {rank}"
    )

    print(
        f"Chunk ID   : "
        f"{result['chunk_id']}"
    )

    print(
        f"RRF Score  : "
        f"{result['rrf_score']:.6f}"
    )

    print(
        f"Reranker   : "
        f"{result['reranker_score']}"
    )


# ============================================================
# VALIDATION
# ============================================================

assert len(results) == 5

print()
print("=" * 60)
print("RERANKER VALIDATION PASSED")
print("=" * 60)