from src.app.azure_openai_client import AzureOpenAIClient
from src.app.prompt_builder import PromptBuilder

from src.document.embedding_service import EmbeddingService
from src.document.vector_indexer import VectorIndexer

from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.rrf_fusion import RRFFusion
from src.retriever.retriever import Retriever
from src.retriever.reranker import BGEReranker

from src.app.rag_chatbot import RAGChatbot


# ============================================================
# EXISTING VECTOR STORE
# ============================================================

FAISS_PATH = (
    "data/vectorstore/test_attention/index.faiss"
)

METADATA_PATH = (
    "data/vectorstore/test_attention/metadata.json"
)


def main():

    print("=" * 70)
    print("RAG CHATBOT TEST")
    print("=" * 70)

    # =========================================================
    # 1. LOAD EXISTING VECTOR STORE
    # =========================================================

    print()
    print("1. Loading existing vector store...")

    indexer = VectorIndexer(
        index_path=FAISS_PATH,
        metadata_path=METADATA_PATH
    )

    indexer.load()

    print(
        f"   Vectors: {indexer.vector_count}"
    )

    if indexer.vector_count == 0:

        raise RuntimeError(
            "FAISS index contains no vectors."
        )

    # =========================================================
    # 2. LOAD EXISTING CHUNKS FROM METADATA
    # =========================================================

    print()
    print("2. Loading existing document chunks...")

    chunks = indexer.metadata

    print(
        f"   Chunks: {len(chunks)}"
    )

    if not chunks:

        raise RuntimeError(
            "No chunk metadata found."
        )

    # =========================================================
    # 3. EMBEDDING SERVICE
    # =========================================================

    print()
    print("3. Loading embedding service...")

    # IMPORTANT:
    # We DO NOT embed the document again.
    #
    # BGE-M3 is only required at query time by
    # the dense retriever.

    embedding = EmbeddingService()

    # =========================================================
    # 4. BM25
    # =========================================================

    print()
    print("4. Creating BM25...")

    bm25 = BM25Retriever(
        chunks
    )

    # =========================================================
    # 5. RRF
    # =========================================================

    print()
    print("5. Creating RRF...")

    rrf = RRFFusion()

    # =========================================================
    # 6. RETRIEVER
    # =========================================================

    print()
    print("6. Creating Retriever...")

    retriever = Retriever(
        indexer=indexer,
        embedding_service=embedding,
        bm25_retriever=bm25,
        rrf_fusion=rrf
    )

    # =========================================================
    # 7. RERANKER
    # =========================================================

    print()
    print("7. Creating BGE Reranker...")

    reranker = BGEReranker()

    # =========================================================
    # 8. PROMPT BUILDER
    # =========================================================

    print()
    print("8. Creating Prompt Builder...")

    prompt_builder = PromptBuilder()

    # =========================================================
    # 9. AZURE OPENAI
    # =========================================================

    print()
    print("9. Creating Azure OpenAI client...")

    openapi_client = AzureOpenAIClient()

    # =========================================================
    # 10. RAG CHATBOT
    # =========================================================

    print()
    print("10. Creating RAG Chatbot...")

    chatbot = RAGChatbot(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        openapi_client=openapi_client,
        chunks=chunks,
    )

    # =========================================================
    # 11. QUESTION
    # =========================================================

    question = (
        "How does scaled dot-product attention work?"
    )

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # =========================================================
    # 12. RUN RAG
    # =========================================================

    print()
    print("=" * 70)
    print("RUNNING RAG CHATBOT")
    print("=" * 70)

    response = chatbot.ask(
        question
    )

    answer = response["answer"]

    selected_results = response[
        "results"
    ]

    sources = response[
        "sources"
    ]

    # =========================================================
    # 13. ANSWER
    # =========================================================

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    # =========================================================
    # 14. SOURCES
    # =========================================================

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in sources:

        print(
            f"• {source.get('chunk_id')} | "
            f"Page {source.get('page_start')}-"
            f"{source.get('page_end')}"
        )

    # =========================================================
    # 15. VALIDATION
    # =========================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if not answer.strip():

        raise RuntimeError(
            "RAGChatbot returned empty answer."
        )

    if not selected_results:

        raise RuntimeError(
            "RAGChatbot selected no context results."
        )

    if len(selected_results) > (
        chatbot.max_context_results
    ):

        raise RuntimeError(
            "Context result limit exceeded."
        )

    if len(sources) != len(
        selected_results
    ):

        raise RuntimeError(
            "Source propagation mismatch."
        )

    print(
        "Answer length :",
        len(answer)
    )

    print(
        "Context used  :",
        len(selected_results)
    )

    print(
        "Sources       :",
        len(sources)
    )

    print()
    print(
        "RAG CHATBOT VALIDATION PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()