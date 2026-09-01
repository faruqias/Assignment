from src.app.azure_openai_client import AzureOpenAIClient
from src.app.prompt_builder import PromptBuilder

from src.document.document_processor import DocumentProcessor
from src.document.document_parser import DocumentParser
from src.document.structure_chunker_new import StructureChunker
from src.document.embedding_service import EmbeddingService
from src.document.vector_indexer import VectorIndexer

from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.rrf_fusion import RRFFusion
from src.retriever.retriever import Retriever
from src.retriever.reranker import BGEReranker

from src.app.rag_chatbot import RAGChatbot


# ============================================================
# TEST DOCUMENT
# ============================================================

PDF_PATH = "data/pdfs/attention.pdf"


# ============================================================
# TEST VECTOR STORE
#
# IMPORTANT:
# This is intentionally separate from the main vector store.
#
# Main:
#     data/vectorstore/
#
# Test:
#     data/vectorstore_test/
#
# Running this test will NOT modify the main database.
# ============================================================

FAISS_PATH = (
    "data/vectorstore_test/index.faiss"
)

METADATA_PATH = (
    "data/vectorstore_test/metadata.json"
)


def main():

    print("=" * 70)
    print("RAG CHATBOT TEST")
    print("=" * 70)

    # =========================================================
    # 1. PROCESS DOCUMENT
    # =========================================================

    print()
    print("1. Processing document...")

    processor = DocumentProcessor()

    result = processor.process(
        PDF_PATH
    )

    # =========================================================
    # 2. PARSE
    # =========================================================

    print()
    print("2. Parsing document...")

    parser = DocumentParser()

    elements = parser.parse(
        result["json_path"]
    )

    print(
        f"   Elements: {len(elements)}"
    )

    if not elements:

        raise RuntimeError(
            "No document elements found."
        )

    # =========================================================
    # 3. STRUCTURE-AWARE CHUNKING
    # =========================================================

    print()
    print("3. Building chunks...")

    chunker = StructureChunker()

    chunks = chunker.chunk(
        elements
    )

    print(
        f"   Chunks: {len(chunks)}"
    )

    if not chunks:

        raise RuntimeError(
            "No chunks generated."
        )

    # =========================================================
    # 4. EMBEDDING SERVICE
    # =========================================================

    print()
    print("4. Loading embedding service...")

    embedding = EmbeddingService()

    # =========================================================
    # 5. GENERATE DOCUMENT EMBEDDINGS
    #
    # This is done ONLY during test ingestion.
    #
    # The chat-only test should NOT execute this step.
    # =========================================================

    print()
    print("5. Generating document embeddings...")

    embeddings = embedding.embed_documents(
        chunks
    )

    print(
        "   Embedding shape:",
        embeddings.shape
    )

    # =========================================================
    # 6. CREATE TEST FAISS INDEX
    # =========================================================

    print()
    print("6. Creating test FAISS index...")

    indexer = VectorIndexer(
        index_path=FAISS_PATH,
        metadata_path=METADATA_PATH
    )

    indexer.index_documents(
        chunks,
        embeddings
    )

    indexer.save()

    print(
        "   Vectors:",
        indexer.vector_count
    )

    print(
        "   FAISS:",
        FAISS_PATH
    )

    print(
        "   Metadata:",
        METADATA_PATH
    )

    # =========================================================
    # 7. BM25
    # =========================================================

    print()
    print("7. Creating BM25...")

    bm25 = BM25Retriever(
        chunks
    )

    # =========================================================
    # 8. RRF
    # =========================================================

    print()
    print("8. Creating RRF...")

    rrf = RRFFusion()

    # =========================================================
    # 9. RETRIEVER
    # =========================================================

    print()
    print("9. Creating Retriever...")

    retriever = Retriever(
        indexer=indexer,
        embedding_service=embedding,
        bm25_retriever=bm25,
        rrf_fusion=rrf
    )

    # =========================================================
    # 10. BGE RERANKER
    #
    # BGEReranker itself reads USE_RERANKER from .env.
    #
    # USE_RERANKER=true
    #     -> BGE Reranker is loaded
    #
    # USE_RERANKER=false
    #     -> Model is NOT loaded
    # =========================================================

    print()
    print("10. Creating BGE Reranker...")

    reranker = BGEReranker()

    # =========================================================
    # 11. PROMPT BUILDER
    # =========================================================

    print()
    print("11. Creating Prompt Builder...")

    prompt_builder = PromptBuilder()

    # =========================================================
    # 12. AZURE OPENAI
    # =========================================================

    print()
    print("12. Creating Azure OpenAI client...")

    openapi_client = AzureOpenAIClient()

    # =========================================================
    # 13. RAG CHATBOT
    # =========================================================

    print()
    print("13. Creating RAG Chatbot...")

    chatbot = RAGChatbot(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        openapi_client=openapi_client,
        chunks=chunks,
    )

    # =========================================================
    # 14. QUESTION
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
    # 15. RUN RAG CHATBOT
    # =========================================================

    print()
    print("=" * 70)
    print("RUNNING RAG CHATBOT")
    print("=" * 70)

    response = chatbot.ask(
        question
    )

    answer = response[
        "answer"
    ]

    selected_results = response[
        "results"
    ]

    sources = response[
        "sources"
    ]

    # =========================================================
    # 16. ANSWER
    # =========================================================

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    # =========================================================
    # 17. SOURCES
    # =========================================================

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    if not sources:

        print(
            "No sources returned."
        )

    for source in sources:

        print(
            f"• {source.get('chunk_id')} | "
            f"Page {source.get('page_start')}-"
            f"{source.get('page_end')}"
        )

    # =========================================================
    # 18. VALIDATION
    # =========================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Answer validation
    # ---------------------------------------------------------

    if not answer or not answer.strip():

        raise RuntimeError(
            "RAGChatbot returned empty answer."
        )

    # ---------------------------------------------------------
    # Context validation
    # ---------------------------------------------------------

    if not selected_results:

        raise RuntimeError(
            "RAGChatbot selected no context results."
        )

    # ---------------------------------------------------------
    # Context result limit
    # ---------------------------------------------------------

    if len(selected_results) > (
        chatbot.max_context_results
    ):

        raise RuntimeError(
            "Context result limit exceeded."
        )

    # ---------------------------------------------------------
    # Source propagation
    # ---------------------------------------------------------

    if len(sources) != len(
        selected_results
    ):

        raise RuntimeError(
            "Source propagation mismatch."
        )

    # ---------------------------------------------------------
    # FAISS validation
    # ---------------------------------------------------------

    if indexer.vector_count != len(
        chunks
    ):

        raise RuntimeError(
            "FAISS vector count does not "
            "match chunk count."
        )

    # =========================================================
    # VALIDATION RESULTS
    # =========================================================

    print(
        "Answer length :",
        len(answer)
    )

    print(
        "Chunks        :",
        len(chunks)
    )

    print(
        "FAISS vectors :",
        indexer.vector_count
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
        "Vector store  :",
        FAISS_PATH
    )

    print(
        "Metadata      :",
        METADATA_PATH
    )

    print()
    print(
        "RAG CHATBOT VALIDATION PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()