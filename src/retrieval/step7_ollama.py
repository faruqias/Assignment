import json
from pathlib import Path

import ollama


# ============================================================
# CONFIG
# ============================================================

CHUNKS_FILE = Path(
    "data/extracted/attention/chunks.json"
)

RERANKED_FILE = Path(
    "data/vectorstore/attention/reranked_results.json"
)

MODEL_NAME = "llama3.2:latest"

TOP_K = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 7 - OLLAMA RAG GENERATION")
print("=" * 70)


with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


with open(
    RERANKED_FILE,
    "r",
    encoding="utf-8"
) as f:

    reranked_results = json.load(f)


print(
    f"Chunks          : {len(chunks)}"
)

print(
    f"Reranked results: {len(reranked_results)}"
)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    query,
    results
):

    context_parts = []


    for rank, result in enumerate(
        results[:TOP_K],
        start=1
    ):

        vector_id = result[
            "vector_id"
        ]

        chunk = chunks[
            vector_id
        ]


        content_type = chunk.get(
            "content_type",
            "text"
        )

        section = chunk.get(
            "section_path",
            []
        )

        page_start = chunk.get(
            "page_start"
        )

        page_end = chunk.get(
            "page_end"
        )

        text = chunk.get(
            "text",
            ""
        )

        caption = chunk.get(
            "caption"
        )


        # ----------------------------------------------------
        # Context header
        # ----------------------------------------------------

        part = []

        part.append(
            f"[Context {rank}]"
        )

        part.append(
            f"Type: {content_type}"
        )

        if page_start is not None:

            part.append(
                f"Page: {page_start}-{page_end}"
            )

        if section:

            part.append(
                "Section: "
                + " > ".join(section)
            )


        # ----------------------------------------------------
        # Figure
        # ----------------------------------------------------

        if content_type == "figure":

            if caption:

                part.append(
                    f"Figure Caption: {caption}"
                )

            if text:

                part.append(
                    f"Figure Text: {text}"
                )


        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        elif content_type == "table":

            if caption:

                part.append(
                    f"Table Caption: {caption}"
                )

            if text:

                part.append(
                    f"Table Content: {text}"
                )


        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        else:

            if text:

                part.append(
                    f"Text: {text}"
                )


        context_parts.append(
            "\n".join(part)
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    query,
    context
):

    return f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information,
   say that the answer is not available in the provided
   context.
4. Give a clear and concise explanation.
5. When relevant, mention the section or page.
6. If a figure or table provides important information,
   explicitly use it in the answer.
7. Do not mention the retrieval process, embeddings,
   FAISS, BM25, RRF, or reranking.

USER QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
""".strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    context
):

    prompt = build_prompt(
        query,
        context
    )


    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        options={
            "temperature": 0.1
        }
    )


    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# RUN QUERIES
# ============================================================

queries = []


for result in reranked_results:

    query = result.get(
        "query"
    )

    if query and query not in queries:

        queries.append(
            query
        )


# ============================================================
# GENERATE ANSWERS
# ============================================================

all_answers = []


for query in queries:

    print()
    print("=" * 70)
    print("QUERY")
    print("=" * 70)

    print(query)


    # --------------------------------------------------------
    # Get results for this query
    # --------------------------------------------------------

    query_results = [

        result

        for result in reranked_results

        if result.get(
            "query"
        ) == query

    ]


    query_results.sort(

        key=lambda x:
        x["reranker_score"],

        reverse=True
    )


    query_results = query_results[
        :TOP_K
    ]


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        query,
        query_results
    )


    print()
    print("Generating answer...")


    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    answer = generate_answer(
        query,
        context
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("ANSWER")
    print("-" * 70)

    print(answer)


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    all_answers.append(
        {
            "query": query,

            "answer": answer,

            "sources": [

                {
                    "chunk_id": result[
                        "chunk_id"
                    ],

                    "content_type": result[
                        "content_type"
                    ],

                    "page_start": result[
                        "page_start"
                    ],

                    "page_end": result[
                        "page_end"
                    ],

                    "section_path": result[
                        "section_path"
                    ],

                    "reranker_score": result[
                        "reranker_score"
                    ]
                }

                for result in query_results
            ]
        }
    )


# ============================================================
# SAVE ANSWERS
# ============================================================

OUTPUT_FILE = Path(
    "data/vectorstore/attention/answers.json"
)


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
        all_answers,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("STEP 7 VALIDATION")
print("=" * 70)


print(
    f"Queries answered: "
    f"{len(all_answers)}"
)


if len(all_answers) != len(queries):

    raise RuntimeError(
        "Not all queries received an answer."
    )


for item in all_answers:

    if not item["answer"].strip():

        raise RuntimeError(
            "Empty answer generated."
        )


print()
print("Validation PASSED")

print(
    f"Output: {OUTPUT_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 7 COMPLETED")
print("=" * 70)