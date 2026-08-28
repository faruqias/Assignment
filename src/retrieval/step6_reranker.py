import json
from pathlib import Path

from FlagEmbedding import FlagReranker


# ============================================================
# CONFIG
# ============================================================

CHUNKS_FILE = Path(
    "data/extracted/attention/chunks.json"
)

RRF_FILE = Path(
    "data/vectorstore/attention/rrf_candidates.json"
)

OUTPUT_FILE = Path(
    "data/vectorstore/attention/reranked_results.json"
)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

TOP_K = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 6 - BGE RERANKER")
print("=" * 70)

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


print(
    f"Chunks        : {len(chunks)}"
)

print(
    f"RRF candidates: {len(candidates)}"
)


# ============================================================
# VALIDATE RRF DATA
# ============================================================

queries = {}

for candidate in candidates:

    query = candidate.get(
        "query"
    )

    if not query:

        raise ValueError(
            "RRF candidate is missing query."
        )

    if query not in queries:

        queries[query] = []

    queries[query].append(
        candidate
    )


print(
    f"Queries       : {len(queries)}"
)

for query, items in queries.items():

    print(
        f"  - {query}"
        f" ({len(items)} candidates)"
    )


# ============================================================
# LOAD RERANKER
# ============================================================

print()
print("Loading BGE Reranker...")
print(
    f"Model: {MODEL_NAME}"
)

reranker = FlagReranker(
    MODEL_NAME,
    use_fp16=False
)

print(
    "BGE Reranker: Ready"
)


# ============================================================
# RERANK EACH QUERY
# ============================================================

all_results = []


for query, query_candidates in queries.items():

    print()
    print("=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print(
        f"\nRRF candidates: "
        f"{len(query_candidates)}"
    )


    # --------------------------------------------------------
    # Prepare pairs
    # --------------------------------------------------------

    pairs = []

    candidate_info = []


    for candidate in query_candidates:

        vector_id = candidate[
            "vector_id"
        ]

        if (
            vector_id < 0
            or vector_id >= len(chunks)
        ):

            raise ValueError(
                f"Invalid vector_id: {vector_id}"
            )


        chunk = chunks[
            vector_id
        ]


        text = chunk.get(
            "text",
            ""
        )


        caption = chunk.get(
            "caption"
        )


        section = chunk.get(
            "section_path",
            []
        )


        # ----------------------------------------------------
        # Build candidate text
        # ----------------------------------------------------

        candidate_text = ""


        if section:

            candidate_text += (
                "Section: "
                + " > ".join(section)
                + "\n"
            )


        if caption:

            candidate_text += (
                "Caption: "
                + caption
                + "\n"
            )


        candidate_text += text


        # ----------------------------------------------------
        # Query + candidate pair
        # ----------------------------------------------------

        pairs.append(
            [
                query,
                candidate_text
            ]
        )


        # ----------------------------------------------------
        # Preserve retrieval information
        # ----------------------------------------------------

        candidate_info.append(
            {
                "query": query,

                "vector_id": vector_id,

                "rrf_score": candidate[
                    "rrf_score"
                ],

                "dense_rank": candidate.get(
                    "dense_rank"
                ),

                "bm25_rank": candidate.get(
                    "bm25_rank"
                )
            }
        )


    # ========================================================
    # RUN RERANKER
    # ========================================================

    print(
        "Running BGE Reranker..."
    )


    scores = reranker.compute_score(
        pairs,
        normalize=True
    )


    if isinstance(
        scores,
        (float, int)
    ):

        scores = [scores]


    # ========================================================
    # CREATE RESULTS
    # ========================================================

    results = []


    for info, score in zip(
        candidate_info,
        scores
    ):

        vector_id = info[
            "vector_id"
        ]

        chunk = chunks[
            vector_id
        ]


        result = {

            "query": info[
                "query"
            ],

            "vector_id": vector_id,

            "chunk_id": chunk.get(
                "chunk_id"
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

            "section_path": chunk.get(
                "section_path",
                []
            ),

            "caption": chunk.get(
                "caption"
            ),

            "dense_rank": info[
                "dense_rank"
            ],

            "bm25_rank": info[
                "bm25_rank"
            ],

            "rrf_score": info[
                "rrf_score"
            ],

            "reranker_score": float(
                score
            )
        }


        results.append(
            result
        )


    # ========================================================
    # SORT BY RERANKER SCORE
    # ========================================================

    results.sort(
        key=lambda x:
            x["reranker_score"],
        reverse=True
    )


    # ========================================================
    # TOP 5
    # ========================================================

    final_results = results[
        :TOP_K
    ]


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print(
        "FINAL TOP 5"
    )

    print(
        "-" * 70
    )


    for rank, result in enumerate(
        final_results,
        start=1
    ):

        print()

        print(
            f"Rank          : {rank}"
        )

        print(
            f"Chunk ID      : "
            f"{result['chunk_id']}"
        )

        print(
            f"Type          : "
            f"{result['content_type']}"
        )

        print(
            f"Page          : "
            f"{result['page_start']}-"
            f"{result['page_end']}"
        )

        print(
            "Section       :",
            " > ".join(
                result[
                    "section_path"
                ]
            )
        )

        print(
            f"Dense Rank    : "
            f"{result['dense_rank']}"
        )

        print(
            f"BM25 Rank     : "
            f"{result['bm25_rank']}"
        )

        print(
            f"RRF Score     : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        if result.get(
            "caption"
        ):

            print(
                "Caption       :",
                result["caption"]
            )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_results.extend(
        final_results
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("STEP 6 VALIDATION")
print("=" * 70)


expected_results = (
    len(queries) * TOP_K
)

actual_results = (
    len(all_results)
)


print(
    f"Queries          : "
    f"{len(queries)}"
)

print(
    f"Expected results : "
    f"{expected_results}"
)

print(
    f"Actual results   : "
    f"{actual_results}"
)


if actual_results != expected_results:

    raise RuntimeError(
        "Unexpected number of reranked results."
    )


# Check query distribution

for query in queries:

    count = sum(
        1
        for result in all_results
        if result["query"] == query
    )

    if count != TOP_K:

        raise RuntimeError(
            f"Expected {TOP_K} results for query "
            f"but got {count}: {query}"
        )


print()
print(
    "Validation PASSED"
)

print(
    f"Top K per query : {TOP_K}"
)

print(
    f"Total results   : {actual_results}"
)

print(
    f"Output          : {OUTPUT_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 6 COMPLETED")
print("=" * 70)