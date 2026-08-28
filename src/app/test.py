import json

from rag_chatbot import (
    RAGChatbot
)


CHUNKS_FILE = (
    "data/extracted/attention/chunks.json"
)


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


# ============================================================
# CREATE SAMPLE RESULTS
# ============================================================

results = []

for chunk in chunks:

    if chunk.get("chunk_id") == (
        "attention_text_6"
    ):

        results.append(
            {
                "vector_id": 7,

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

                "text": chunk.get(
                    "text",
                    ""
                ),

                "reranker_score": 0.9953
            }
        )

        break


# ============================================================
# CREATE CHATBOT
# ============================================================

chatbot = RAGChatbot()


# ============================================================
# TEST
# ============================================================

question = (
    "How does scaled dot-product attention work?"
)


print()
print("=" * 60)
print("RAG CHATBOT TEST")
print("=" * 60)

print(
    f"Question: {question}"
)

print()
print("Generating answer...")
print()


answer = ""

for token in chatbot.stream(
    question,
    results
):

    answer += token

    print(
        token,
        end="",
        flush=True
    )

print()


# ============================================================
# SOURCES
# ============================================================

sources = chatbot.build_sources(
    results
)

print("Sources:")
print(sources)


# ============================================================
# VALIDATION
# ============================================================

if not answer.strip():

    raise RuntimeError(
        "Ollama returned an empty answer."
    )

print()
print(
    "Generated answer length:",
    len(answer)
)

print()
print("=" * 60)
print("RAG CHATBOT VALIDATION PASSED")
print("=" * 60)