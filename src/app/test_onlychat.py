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
    "data/vectorstore/index.faiss"
)

METADATA_PATH = (
    "data/vectorstore/metadata.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAG CHATBOT - CONVERSATIONAL MEMORY TEST")
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
    # 2. LOAD CHUNKS
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
    #
    # We are NOT embedding the PDF again.
    #
    # BGE-M3 is only used for the
    # question/query embedding.

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
    # 11. CLEAR MEMORY
    # =========================================================

    print()
    print("11. Clearing conversation memory...")

    chatbot.memory.clear()

    print(
        f"   Memory interactions: "
        f"{chatbot.memory.count()}"
    )

    if chatbot.memory.count() != 0:

        raise RuntimeError(
            "Memory was not cleared."
        )

    # =========================================================
    # 12. CONVERSATION QUESTIONS
    # =========================================================

    questions = [

        # -----------------------------------------------------
        # Interaction 1
        # -----------------------------------------------------

        "What is BERT?",

        # -----------------------------------------------------
        # Interaction 2
        # -----------------------------------------------------

        "What is its pre-training objective?",

        # -----------------------------------------------------
        # Interaction 3
        # -----------------------------------------------------

        "What are the main contributions of BERT?",

        # -----------------------------------------------------
        # Interaction 4
        # -----------------------------------------------------

        "What datasets were used to evaluate BERT?",

        # -----------------------------------------------------
        # Interaction 5
        # -----------------------------------------------------

        "What experimental results support its effectiveness?"
    ]

    # =========================================================
    # 13. RUN CONVERSATION
    # =========================================================

    answers = []

    print()
    print("=" * 70)
    print("RUNNING 5-INTERACTION CONVERSATION")
    print("=" * 70)

    for number, question in enumerate(
        questions,
        start=1
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"INTERACTION {number}"
        )

        print(
            "=" * 70
        )

        print()
        print(
            "QUESTION:"
        )

        print(question)

        # -----------------------------------------------------
        # Ask chatbot
        # -----------------------------------------------------

        response = chatbot.ask(
            question
        )

        answer = response.get(
            "answer",
            ""
        )

        answers.append(
            answer
        )

        # -----------------------------------------------------
        # Results
        # -----------------------------------------------------

        selected_results = response.get(
            "results",
            []
        )

        sources = response.get(
            "sources",
            []
        )

        print()
        print(
            "ANSWER:"
        )

        print(answer)

        # -----------------------------------------------------
        # Sources
        # -----------------------------------------------------

        print()
        print(
            "SOURCES:"
        )

        for source in sources:

            print(
                f"• {source.get('chunk_id')} | "
                f"Page "
                f"{source.get('page_start')}-"
                f"{source.get('page_end')}"
            )

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        memory_count = (
            chatbot.memory.count()
        )

        print()
        print(
            f"MEMORY COUNT: "
            f"{memory_count}"
        )

        # -----------------------------------------------------
        # Validate answer
        # -----------------------------------------------------

        if not answer.strip():

            raise RuntimeError(
                f"Interaction {number} "
                "returned an empty answer."
            )

        # -----------------------------------------------------
        # Validate context
        # -----------------------------------------------------

        if not selected_results:

            raise RuntimeError(
                f"Interaction {number} "
                "selected no context."
            )

        # -----------------------------------------------------
        # Validate sources
        # -----------------------------------------------------

        if len(sources) != len(
            selected_results
        ):

            raise RuntimeError(
                f"Interaction {number}: "
                "source propagation mismatch."
            )

        # -----------------------------------------------------
        # Validate memory limit
        # -----------------------------------------------------

        if memory_count > 4:

            raise RuntimeError(
                "Conversation memory exceeded "
                "the maximum of 4 interactions."
            )

    # =========================================================
    # 14. DISPLAY FINAL MEMORY
    # =========================================================

    print()
    print("=" * 70)
    print("FINAL CONVERSATION MEMORY")
    print("=" * 70)

    memory = (
        chatbot.memory.get_history()
    )

    for number, interaction in enumerate(
        memory,
        start=1
    ):

        print()
        print(
            f"Memory Interaction {number}"
        )

        print(
            "Question:"
        )

        print(
            interaction["question"]
        )

        print(
            "Answer:"
        )

        print(
            interaction["answer"]
        )

    # =========================================================
    # 15. MEMORY VALIDATION
    # =========================================================

    print()
    print("=" * 70)
    print("MEMORY VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Exactly 4 interactions should remain
    # ---------------------------------------------------------

    if len(memory) != 4:

        raise RuntimeError(
            "Memory validation failed. "
            f"Expected 4 interactions, "
            f"found {len(memory)}."
        )

    # ---------------------------------------------------------
    # First interaction should have been removed
    # ---------------------------------------------------------

    first_question = (
        memory[0]["question"]
    )

    if first_question == questions[0]:

        raise RuntimeError(
            "Oldest interaction was not removed."
        )

    # ---------------------------------------------------------
    # Expected questions are Q2-Q5
    # ---------------------------------------------------------

    expected_questions = (
        questions[1:]
    )

    actual_questions = [
        interaction["question"]
        for interaction in memory
    ]

    if actual_questions != (
        expected_questions
    ):

        raise RuntimeError(
            "Memory ordering/content mismatch."
            f"\nExpected: {expected_questions}"
            f"\nActual:   {actual_questions}"
        )

    # =========================================================
    # 16. FINAL VALIDATION
    # =========================================================

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        "Interactions executed :",
        len(questions)
    )

    print(
        "Interactions retained :",
        len(memory)
    )

    print(
        "Oldest interaction    :",
        questions[0]
    )

    print(
        "Oldest retained       :",
        memory[0]["question"]
    )

    print()
    print(
        "Conversation memory is limited "
        "to the last 4 interactions."
    )

    print()
    print(
        "RAG CHATBOT MEMORY VALIDATION PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()