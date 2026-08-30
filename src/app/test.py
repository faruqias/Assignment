from src.app.ollama_client import OllamaClient
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


PDF_PATH = "data/pdfs/attention.pdf"

FAISS_PATH = "data/vectorstore/test_attention/index.faiss"
METADATA_PATH = "data/vectorstore/test_attention/metadata.json"


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

    # =========================================================
    # 3. CHUNK
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
    # 5. GENERATE EMBEDDINGS
    # =========================================================

    print()
    print("5. Generating embeddings...")

    embeddings = embedding.embed_documents(
        chunks
    )

    print(
        "   Embedding shape:",
        embeddings.shape
    )

    # =========================================================
    # 6. VECTOR INDEX
    # =========================================================

    print()
    print("6. Creating FAISS index...")

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
    # 10. RERANKER
    # =========================================================

    print()
    print("10. Creating BGE Reranker...")

    reranker = BGEReranker()

    # =========================================================
    # 11. CHATBOT
    # =========================================================

    print()
    print("11. Creating RAG Chatbot...")

    prompt_builder = PromptBuilder()

    llm_client = OllamaClient()

    chatbot = RAGChatbot(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        chunks=chunks,
    )

    # =========================================================
    # 12. QUESTION
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
    # 13. RUN RAG CHATBOT
    # =========================================================

    print()
    print("=" * 70)
    print("RUNNING RAG CHATBOT")
    print("=" * 70)

    response = chatbot.ask(
        question
    )

    answer = response["answer"]

    selected_results = response["results"]

    sources = response["sources"]

    # =========================================================
    # 14. ANSWER
    # =========================================================

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    # =========================================================
    # 15. SOURCES
    # =========================================================

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in sources:

        print(
            f"• {source['chunk_id']} | "
            f"Page {source['page_start']}-"
            f"{source['page_end']}"
        )

    # =========================================================
    # 16. VALIDATION
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

    if len(selected_results) > chatbot.max_context_results:

        raise RuntimeError(
            "Context result limit exceeded."
        )

    if len(sources) != len(selected_results):

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