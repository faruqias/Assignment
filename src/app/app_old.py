import json
import time
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import ollama

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    BASE_DIR
    / "data"
    / "extracted"
    / "attention"
    / "chunks.json"
)

FAISS_FILE = (
    BASE_DIR
    / "data"
    / "vectorstore"
    / "attention"
    / "index.faiss"
)

METADATA_FILE = (
    BASE_DIR
    / "data"
    / "vectorstore"
    / "attention"
    / "metadata.json"
)

EMBEDDING_MODEL = "BAAI/bge-m3"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_MODEL = "llama3.2:latest"

DENSE_TOP_K = 10
BM25_TOP_K = 10
RRF_TOP_K = 5
FINAL_TOP_K = 3

RRF_K = 60

OLLAMA_MAX_TOKENS = 180


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 70)
print("INITIALIZING FINAL RAG APPLICATION")
print("=" * 70)


# ============================================================
# LOAD CHUNKS
# ============================================================

print()
print("1. Loading chunks...")

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(
    f"   Chunks loaded: {len(chunks)}"
)


# ============================================================
# LOAD FAISS
# ============================================================

print()
print("2. Loading FAISS...")

index = faiss.read_index(
    str(FAISS_FILE)
)

print(
    f"   FAISS vectors: {index.ntotal}"
)


# ============================================================
# LOAD METADATA
# ============================================================

print()
print("3. Loading metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(
    "   Metadata loaded."
)


# ============================================================
# BUILD BM25
# ============================================================

print()
print("4. Building BM25...")


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
    f"   BM25 documents: "
    f"{len(bm25_corpus)}"
)


# ============================================================
# LOAD BGE-M3
# ============================================================

print()
print("5. Loading BGE-M3...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "   BGE-M3 loaded."
)


# ============================================================
# LOAD BGE RERANKER
# ============================================================

print()
print("6. Loading BGE Reranker...")

print(
    f"   Model: {RERANKER_MODEL}"
)

reranker = FlagReranker(
    RERANKER_MODEL,
    use_fp16=False
)

print(
    "   BGE Reranker loaded."
)


# ============================================================
# OLLAMA
# ============================================================

print()
print("7. Ollama model:")

print(
    f"   {OLLAMA_MODEL}"
)


print()
print("=" * 70)
print("RAG INITIALIZATION COMPLETE")
print("=" * 70)


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(query):

    query_embedding = embedding_model.encode(

        [query],

        normalize_embeddings=True,

        convert_to_numpy=True
    )

    query_embedding = (
        query_embedding
        .astype(np.float32)
    )

    scores, ids = index.search(

        query_embedding,

        DENSE_TOP_K
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

def bm25_search(query):

    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked_ids = np.argsort(
        scores
    )[::-1]

    results = []

    for rank, vector_id in enumerate(

        ranked_ids[
            :BM25_TOP_K
        ],

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
# RRF
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results
):

    fused = {}

    dense_ranks = {}

    bm25_ranks = {}


    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    for result in dense_results:

        vector_id = result[
            "vector_id"
        ]

        rank = result[
            "rank"
        ]

        dense_ranks[
            vector_id
        ] = rank

        fused[
            vector_id
        ] = (

            fused.get(
                vector_id,
                0.0
            )

            +

            1.0 / (
                RRF_K + rank
            )
        )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    for result in bm25_results:

        vector_id = result[
            "vector_id"
        ]

        rank = result[
            "rank"
        ]

        bm25_ranks[
            vector_id
        ] = rank

        fused[
            vector_id
        ] = (

            fused.get(
                vector_id,
                0.0
            )

            +

            1.0 / (
                RRF_K + rank
            )
        )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked = sorted(

        fused.items(),

        key=lambda x: x[1],

        reverse=True
    )


    results = []


    for vector_id, score in ranked[
        :RRF_TOP_K
    ]:

        results.append(
            {
                "vector_id": int(
                    vector_id
                ),

                "rrf_score": float(
                    score
                ),

                "dense_rank":
                    dense_ranks.get(
                        vector_id
                    ),

                "bm25_rank":
                    bm25_ranks.get(
                        vector_id
                    )
            }
        )


    return results


# ============================================================
# BUILD RERANKER TEXT
# ============================================================

def build_candidate_text(chunk):

    parts = []

    section = chunk.get(
        "section_path",
        []
    )

    caption = chunk.get(
        "caption"
    )

    text = chunk.get(
        "text",
        ""
    )


    if section:

        parts.append(
            "Section: "
            +
            " > ".join(section)
        )


    if caption:

        parts.append(
            "Caption: "
            +
            caption
        )


    if text:

        parts.append(
            text
        )


    return "\n".join(
        parts
    )


# ============================================================
# RERANK
# ============================================================

def rerank(
    query,
    candidates
):

    pairs = []


    for candidate in candidates:

        vector_id = candidate[
            "vector_id"
        ]

        chunk = chunks[
            vector_id
        ]

        candidate_text = (
            build_candidate_text(
                chunk
            )
        )

        pairs.append(
            [
                query,
                candidate_text
            ]
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


    results = []


    for candidate, score in zip(

        candidates,

        scores
    ):

        vector_id = candidate[
            "vector_id"
        ]

        chunk = chunks[
            vector_id
        ]


        results.append(
            {
                "vector_id":
                    vector_id,

                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "content_type":
                    chunk.get(
                        "content_type"
                    ),

                "page_start":
                    chunk.get(
                        "page_start"
                    ),

                "page_end":
                    chunk.get(
                        "page_end"
                    ),

                "section_path":
                    chunk.get(
                        "section_path",
                        []
                    ),

                "caption":
                    chunk.get(
                        "caption"
                    ),

                "text":
                    chunk.get(
                        "text",
                        ""
                    ),

                "dense_rank":
                    candidate.get(
                        "dense_rank"
                    ),

                "bm25_rank":
                    candidate.get(
                        "bm25_rank"
                    ),

                "rrf_score":
                    candidate.get(
                        "rrf_score"
                    ),

                "reranker_score":
                    float(score)
            }
        )


    results.sort(

        key=lambda x:
        x["reranker_score"],

        reverse=True
    )


    return results[
        :FINAL_TOP_K
    ]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context = []


    for rank, result in enumerate(

        results,

        start=1
    ):

        part = []


        part.append(
            f"[Source {rank}]"
        )


        part.append(
            f"Type: "
            f"{result['content_type']}"
        )


        if (
            result["page_start"]
            is not None
        ):

            part.append(
                f"Page: "
                f"{result['page_start']}-"
                f"{result['page_end']}"
            )


        if result["section_path"]:

            part.append(
                "Section: "
                +
                " > ".join(
                    result[
                        "section_path"
                    ]
                )
            )


        if result["caption"]:

            part.append(
                "Caption: "
                +
                result["caption"]
            )


        if result["text"]:

            part.append(
                "Content: "
                +
                result["text"]
            )


        context.append(
            "\n".join(part)
        )


    return "\n\n".join(
        context
    )


# ============================================================
# OLLAMA PROMPT
# ============================================================

def build_prompt(
    query,
    context
):

    return f"""
You are a document question-answering assistant.

Answer the question using ONLY the document context.

Rules:

- Do not use outside knowledge.
- Do not invent facts.
- If the document does not contain the answer, say so.
- Be concise and directly answer the question.
- Prefer 2-4 short paragraphs or a short bullet list.
- Use figures and tables when relevant.
- Mention the relevant page or section.
- Do not mention FAISS, BM25, RRF, embeddings,
  reranking, or the RAG pipeline.

QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

ANSWER:
""".strip()


# ============================================================
# SOURCE DISPLAY
# ============================================================

def build_sources(results):

    sources = []


    for rank, result in enumerate(

        results,

        start=1
    ):

        source = (
            f"**{rank}. "
            f"{result['chunk_id']}**"
        )


        source += (
            f"\n\n"
            f"- Type: "
            f"{result['content_type']}"
        )


        source += (
            f"\n"
            f"- Page: "
            f"{result['page_start']}-"
            f"{result['page_end']}"
        )


        if result[
            "section_path"
        ]:

            source += (
                "\n"
                "- Section: "
                +
                " > ".join(
                    result[
                        "section_path"
                    ]
                )
            )


        if result[
            "content_type"
        ] == "figure":

            if result[
                "caption"
            ]:

                source += (
                    "\n"
                    "- Figure: "
                    +
                    result[
                        "caption"
                    ]
                )


        if result[
            "content_type"
        ] == "table":

            if result[
                "caption"
            ]:

                source += (
                    "\n"
                    "- Table: "
                    +
                    result[
                        "caption"
                    ]
                )


        sources.append(
            source
        )


    return "\n\n".join(
        sources
    )


# ============================================================
# RAG CHAT
# ============================================================

def rag_chat(message, history):

    print()
    print("=" * 70)
    print("NEW RAG QUERY")
    print("=" * 70)

    print(f"Query: {message}")

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not message or not message.strip():

        yield (
            history or [],
            "⚠️ Please enter a question."
        )

        return

    # --------------------------------------------------------
    # Initialize history
    # --------------------------------------------------------

    if history is None:
        history = []

    history = list(history)

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    # --------------------------------------------------------
    # Add assistant placeholder
    # --------------------------------------------------------

    history.append(
        {
            "role": "assistant",
            "content": "🔎 Searching the document..."
        }
    )

    yield (
        history,
        "🔎 Searching the document..."
    )

    # ========================================================
    # 1. BGE-M3 + FAISS
    # ========================================================

    start = time.perf_counter()

    dense_results = dense_search(message)

    dense_time = (
        time.perf_counter() - start
    )

    print(
        f"1. BGE-M3 + FAISS : "
        f"{dense_time:.3f} sec"
    )

    history[-1] = {
        "role": "assistant",
        "content": "🔎 Running keyword search..."
    }

    yield (
        history,
        "🔎 Running keyword search..."
    )

    # ========================================================
    # 2. BM25
    # ========================================================

    start = time.perf_counter()

    bm25_results = bm25_search(message)

    bm25_time = (
        time.perf_counter() - start
    )

    print(
        f"2. BM25           : "
        f"{bm25_time:.3f} sec"
    )

    # ========================================================
    # 3. RRF
    # ========================================================

    start = time.perf_counter()

    rrf_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    rrf_time = (
        time.perf_counter() - start
    )

    print(
        f"3. RRF             : "
        f"{rrf_time:.3f} sec"
    )

    history[-1] = {
        "role": "assistant",
        "content": "🎯 Reranking relevant passages..."
    }

    yield (
        history,
        "🎯 Reranking relevant passages..."
    )

    # ========================================================
    # 4. BGE RERANKER
    # ========================================================

    start = time.perf_counter()

    final_results = rerank(
        message,
        rrf_results
    )

    reranker_time = (
        time.perf_counter() - start
    )

    print(
        f"4. BGE Reranker    : "
        f"{reranker_time:.3f} sec"
    )

    # ========================================================
    # 5. CONTEXT
    # ========================================================

    start = time.perf_counter()

    context = build_context(
        final_results
    )

    context_time = (
        time.perf_counter() - start
    )

    print(
        f"5. Context build   : "
        f"{context_time:.3f} sec"
    )

    # ========================================================
    # 6. PROMPT
    # ========================================================

    prompt = build_prompt(
        message,
        context
    )

    # ========================================================
    # 7. OLLAMA
    # ========================================================

    history[-1] = {
        "role": "assistant",
        "content": "🤖 Generating answer..."
    }

    yield (
        history,
        "🤖 Generating answer..."
    )

    print()
    print("6. Sending context to Ollama...")
    print(f"   Model: {OLLAMA_MODEL}")

    start = time.perf_counter()

    stream = ollama.chat(
        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        stream=True,

        options={
            "temperature": 0.1,
            "num_predict": OLLAMA_MAX_TOKENS
        }
    )

    answer = ""

    first_token_time = None

    for response in stream:

        if first_token_time is None:

            first_token_time = (
                time.perf_counter() - start
            )

            print(
                f"   First token     : "
                f"{first_token_time:.3f} sec"
            )

        token = response[
            "message"
        ].get(
            "content",
            ""
        )

        answer += token

        history[-1] = {
            "role": "assistant",
            "content": answer
        }

        yield (
            history,
            "🤖 Generating answer..."
        )

    ollama_time = (
        time.perf_counter() - start
    )

    print(
        f"6. Ollama total    : "
        f"{ollama_time:.3f} sec"
    )

    # ========================================================
    # 8. SOURCES
    # ========================================================

    start = time.perf_counter()

    sources = build_sources(
        final_results
    )

    sources_time = (
        time.perf_counter() - start
    )

    print(
        f"7. Sources         : "
        f"{sources_time:.3f} sec"
    )

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    final_answer = (
        answer
        + "\n\n---\n\n"
        + "### 📚 Sources\n\n"
        + sources
    )

    history[-1] = {
        "role": "assistant",
        "content": final_answer
    }

    # ========================================================
    # PERFORMANCE
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 70)
    print("PERFORMANCE")
    print("=" * 70)

    print(
        f"BGE-M3 + FAISS : "
        f"{dense_time:.3f} sec"
    )

    print(
        f"BM25           : "
        f"{bm25_time:.3f} sec"
    )

    print(
        f"RRF            : "
        f"{rrf_time:.3f} sec"
    )

    print(
        f"BGE Reranker   : "
        f"{reranker_time:.3f} sec"
    )

    print(
        f"Context build  : "
        f"{context_time:.3f} sec"
    )

    print(
        f"Ollama         : "
        f"{ollama_time:.3f} sec"
    )

    print(
        f"Sources        : "
        f"{sources_time:.3f} sec"
    )

    print("-" * 70)

    print(
        f"TOTAL          : "
        f"{total_time:.3f} sec"
    )

    if first_token_time is not None:

        print(
            f"FIRST TOKEN    : "
            f"{first_token_time:.3f} sec"
        )

    print("=" * 70)

    yield (
        history,
        "✅ Answer complete"
    )

def rag_chat_remove(
    message,
    history
):

    print()
    print("=" * 70)
    print("NEW RAG QUERY")
    print("=" * 70)

    print(
        f"Query: {message}"
    )


    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    if not message.strip():

        yield (
            history or [],
            "⚠️ Please enter a question."
        )

        return


    # ========================================================
    # INITIALIZE HISTORY
    # ========================================================

    if history is None:

        history = []


    history = list(history)


    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    history.append(
        [
            message,
            ""
        ]
    )


    # ========================================================
    # 1. BGE-M3 + FAISS
    # ========================================================

    yield (
        history,
        "🔎 Searching the document..."
    )


    start = time.perf_counter()


    dense_results = dense_search(
        message
    )


    dense_time = (
        time.perf_counter()
        - start
    )


    print(
        f"1. BGE-M3 + FAISS : "
        f"{dense_time:.3f} sec"
    )


    # ========================================================
    # 2. BM25
    # ========================================================

    yield (
        history,
        "🔎 Running keyword search..."
    )


    start = time.perf_counter()


    bm25_results = bm25_search(
        message
    )


    bm25_time = (
        time.perf_counter()
        - start
    )


    print(
        f"2. BM25           : "
        f"{bm25_time:.3f} sec"
    )


    # ========================================================
    # 3. RRF
    # ========================================================

    start = time.perf_counter()


    rrf_results = reciprocal_rank_fusion(

        dense_results,

        bm25_results
    )


    rrf_time = (
        time.perf_counter()
        - start
    )


    print(
        f"3. RRF            : "
        f"{rrf_time:.3f} sec"
    )


    # ========================================================
    # 4. BGE RERANKER
    # ========================================================

    yield (
        history,
        "🎯 Reranking relevant passages..."
    )


    start = time.perf_counter()


    final_results = rerank(

        message,

        rrf_results
    )


    reranker_time = (
        time.perf_counter()
        - start
    )


    print(
        f"4. BGE Reranker    : "
        f"{reranker_time:.3f} sec"
    )


    # ========================================================
    # 5. CONTEXT
    # ========================================================

    start = time.perf_counter()


    context = build_context(
        final_results
    )


    context_time = (
        time.perf_counter()
        - start
    )


    print(
        f"5. Context build   : "
        f"{context_time:.3f} sec"
    )


    # ========================================================
    # 6. PROMPT
    # ========================================================

    prompt = build_prompt(

        message,

        context
    )


    # ========================================================
    # 7. OLLAMA
    # ========================================================

    yield (
        history,
        "🤖 Generating answer..."
    )


    print()
    print(
        "6. Sending context to Ollama..."
    )

    print(
        f"   Model: {OLLAMA_MODEL}"
    )


    start = time.perf_counter()


    stream = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        stream=True,

        options={
            "temperature": 0.1,
            "num_predict":
                OLLAMA_MAX_TOKENS
        }
    )


    answer = ""

    first_token_time = None


    for response in stream:

        if first_token_time is None:

            first_token_time = (
                time.perf_counter()
                - start
            )

            print(
                f"   First token     : "
                f"{first_token_time:.3f} sec"
            )


        token = response[
            "message"
        ].get(
            "content",
            ""
        )


        answer += token


        # ----------------------------------------------------
        # Update assistant response
        # ----------------------------------------------------

        history[-1][1] = answer


        yield (
            history,
            "🤖 Generating answer..."
        )


    ollama_time = (
        time.perf_counter()
        - start
    )


    print(
        f"6. Ollama total    : "
        f"{ollama_time:.3f} sec"
    )


    # ========================================================
    # 8. SOURCES
    # ========================================================

    start = time.perf_counter()


    sources = build_sources(
        final_results
    )


    sources_time = (
        time.perf_counter()
        - start
    )


    print(
        f"7. Sources         : "
        f"{sources_time:.3f} sec"
    )


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    final_answer = (

        answer

        +

        "\n\n---\n\n"

        +

        "### 📚 Sources\n\n"

        +

        sources
    )


    history[-1][1] = final_answer


    # ========================================================
    # PERFORMANCE
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )


    print()
    print("=" * 70)
    print("PERFORMANCE")
    print("=" * 70)


    print(
        f"BGE-M3 + FAISS : "
        f"{dense_time:.3f} sec"
    )


    print(
        f"BM25           : "
        f"{bm25_time:.3f} sec"
    )


    print(
        f"RRF            : "
        f"{rrf_time:.3f} sec"
    )


    print(
        f"BGE Reranker   : "
        f"{reranker_time:.3f} sec"
    )


    print(
        f"Context build  : "
        f"{context_time:.3f} sec"
    )


    print(
        f"Ollama         : "
        f"{ollama_time:.3f} sec"
    )


    print(
        f"Sources         : "
        f"{sources_time:.3f} sec"
    )


    print(
        "-" * 70
    )


    print(
        f"TOTAL           : "
        f"{total_time:.3f} sec"
    )


    if first_token_time is not None:

        print(
            f"FIRST TOKEN     : "
            f"{first_token_time:.3f} sec"
        )


    print("=" * 70)


    # ========================================================
    # COMPLETE
    # ========================================================

    yield (
        history,
        "✅ Answer complete"
    )


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():

    return [], "🟢 Ready"


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(

    title="Document RAG Assistant"

) as demo:


    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        """
# 📚 Document RAG Assistant

Ask questions about the loaded research document.

**Docling → BGE-M3 → FAISS + BM25 → RRF → BGE Reranker → Ollama**
"""
    )


    # ========================================================
    # CHATBOT
    # ========================================================

    chatbot = gr.Chatbot(
    label="Conversation",
    height=600
)


    # ========================================================
    # STATUS
    # ========================================================

    status = gr.Markdown(
        "🟢 Ready"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with gr.Row():

        message = gr.Textbox(

            placeholder=(
                "Ask something about "
                "the document..."
            ),

            lines=2,

            scale=8,

            show_label=False
        )


        send = gr.Button(

            "➤",

            variant="primary",

            scale=1
        )


    # ========================================================
    # CONTROLS
    # ========================================================

    with gr.Row():

        clear = gr.Button(
            "🗑 Clear Chat"
        )


    # ========================================================
    # EVENTS
    # ========================================================

    send.click(

        fn=rag_chat,

        inputs=[
            message,
            chatbot
        ],

        outputs=[
            chatbot,
            status
        ]

    ).then(

        fn=lambda: "",

        inputs=None,

        outputs=message
    )


    message.submit(

        fn=rag_chat,

        inputs=[
            message,
            chatbot
        ],

        outputs=[
            chatbot,
            status
        ]

    ).then(

        fn=lambda: "",

        inputs=None,

        outputs=message
    )


    clear.click(

        fn=clear_chat,

        inputs=None,

        outputs=[
            chatbot,
            status
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    demo.launch()