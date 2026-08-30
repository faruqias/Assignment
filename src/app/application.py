from pathlib import Path

from src.document.document_processor import DocumentProcessor
from src.document.document_parser import DocumentParser
from src.document.structure_chunker_new import StructureChunker
from src.document.embedding_service import EmbeddingService
from src.document.vector_indexer import VectorIndexer

from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.rrf_fusion import RRFFusion
from src.retriever.retriever import Retriever
from src.retriever.reranker import BGEReranker

from src.app.prompt_builder import PromptBuilder
from src.app.rag_chatbot import RAGChatbot

import ollama


class OllamaClient:
    """
    Small adapter around the Ollama Python client.
    """

    def generate(
        self,
        prompt,
        model,
        temperature,
        max_tokens,
    ):

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        return response["message"]["content"]


class RAGApplication:
    """
    Application-level orchestration for the RAG system.

    Responsibilities:
        - Process uploaded PDFs
        - Parse documents
        - Create chunks
        - Generate embeddings
        - Build retrieval indexes
        - Create RAG chatbot
        - Handle UI requests
    """

    VECTORSTORE_ROOT = Path(
        "data/vectorstore"
    )

    def __init__(self):

        print()
        print("=" * 70)
        print("INITIALIZING RAG APPLICATION")
        print("=" * 70)

        # =====================================================
        # 1. DOCUMENT PROCESSOR
        # =====================================================

        print()
        print("1. Document Processor")

        self.processor = DocumentProcessor()

        # =====================================================
        # 2. DOCUMENT PARSER
        # =====================================================

        print()
        print("2. Document Parser")

        self.parser = DocumentParser()

        # =====================================================
        # 3. STRUCTURE CHUNKER
        # =====================================================

        print()
        print("3. Structure Chunker")

        self.chunker = StructureChunker()

        # =====================================================
        # 4. EMBEDDING SERVICE
        # =====================================================

        print()
        print("4. Embedding Service")

        self.embedding = EmbeddingService()

        # =====================================================
        # 5. VECTOR INDEXER
        # =====================================================

        print()
        print("5. Vector Indexer")

        # Created per document during upload.
        self.indexer = None

        # =====================================================
        # 6. RERANKER
        # =====================================================

        print()
        print("6. BGE Reranker")

        self.reranker = BGEReranker()

        # =====================================================
        # 7. PROMPT BUILDER
        # =====================================================

        print()
        print("7. Prompt Builder")

        self.prompt_builder = PromptBuilder()

        # =====================================================
        # 8. OLLAMA CLIENT
        # =====================================================

        print()
        print("8. Ollama Client")

        self.llm_client = OllamaClient()

        # =====================================================
        # CHATBOT
        # =====================================================

        self.chatbot = None

        # =====================================================
        # DOCUMENT STATE
        # =====================================================

        self.documents = {}

        print()
        print("=" * 70)
        print("RAG APPLICATION READY")
        print("=" * 70)

    # =========================================================
    # DOCUMENT UPLOAD
    # =========================================================

    def upload_documents(self, files):
        """
        Process and index uploaded PDF documents.

        Current implementation processes each PDF and creates
        its own vector store.

        Returns:
            str: UI status message
        """

        if not files:

            return "Please upload at least one PDF document."

        if not isinstance(files, list):

            files = [files]

        messages = []

        for file_path in files:

            try:

                file_path = str(file_path)

                document_name = Path(
                    file_path
                ).name

                document_id = Path(
                    file_path
                ).stem

                print()
                print("=" * 60)
                print(
                    f"PROCESSING: {document_name}"
                )
                print("=" * 60)

                # =================================================
                # 1. PROCESS DOCUMENT
                # =================================================

                result = self.processor.process(
                    file_path
                )

                # =================================================
                # 2. PARSE
                # =================================================

                elements = self.parser.parse(
                    result["json_path"]
                )

                # =================================================
                # 3. CHUNK
                # =================================================

                chunks = self.chunker.chunk(
                    elements
                )

                if not chunks:

                    messages.append(
                        f"{document_name}: no chunks generated."
                    )

                    continue

                # =================================================
                # 4. METADATA
                # =================================================

                for chunk in chunks:

                    chunk["document_id"] = (
                        document_id
                    )

                    chunk["document_name"] = (
                        document_name
                    )

                # =================================================
                # 5. EMBEDDINGS
                # =================================================

                embeddings = (
                    self.embedding.embed_documents(
                        chunks
                    )
                )

                # =================================================
                # 6. VECTOR INDEX
                # =================================================

                vectorstore_path = (
                    self.VECTORSTORE_ROOT
                    / document_id
                )

                vectorstore_path.mkdir(
                    parents=True,
                    exist_ok=True
                )

                index_path = (
                    vectorstore_path
                    / "index.faiss"
                )

                metadata_path = (
                    vectorstore_path
                    / "metadata.json"
                )

                indexer = VectorIndexer(
                    index_path=str(index_path),
                    metadata_path=str(metadata_path)
                )

                indexer.index_documents(
                    chunks,
                    embeddings
                )

                indexer.save()

                # =================================================
                # 7. BM25
                # =================================================

                bm25 = BM25Retriever(
                    chunks
                )

                # =================================================
                # 8. RRF
                # =================================================

                rrf = RRFFusion()

                # =================================================
                # 9. RETRIEVER
                # =================================================

                retriever = Retriever(
                    indexer=indexer,
                    embedding_service=self.embedding,
                    bm25_retriever=bm25,
                    rrf_fusion=rrf
                )

                # =================================================
                # SAVE DOCUMENT
                # =================================================

                self.documents[document_id] = {
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunks": chunks,
                    "indexer": indexer,
                    "bm25": bm25,
                    "rrf": rrf,
                    "retriever": retriever,
                }

                messages.append(
                    f"✓ {document_name} "
                    f"({len(chunks)} chunks)"
                )

            except Exception as exc:

                messages.append(
                    f"✗ {document_name}: {exc}"
                )

                print()
                print(
                    f"ERROR processing {document_name}:"
                )
                print(exc)

        # =========================================================
        # BUILD ACTIVE CHATBOT
        # =========================================================

        if self.documents:

            # For now use the latest uploaded document.
            latest_document = list(
                self.documents.values()
            )[-1]

            self.indexer = (
                latest_document["indexer"]
            )

            self.chatbot = RAGChatbot(
                retriever=latest_document["retriever"],
                reranker=self.reranker,
                prompt_builder=self.prompt_builder,
                llm_client=self.llm_client,
                chunks=chunks
            )

        return "\n".join(messages)

    # =========================================================
    # ASK QUESTION
    # =========================================================

    def ask_question(
        self,
        message,
        history
    ):
        """
        Ask a question through the RAG chatbot.

        Gradio 6.x expects message dictionaries.
        """

        if not message or not message.strip():

            return history or []

        if self.chatbot is None:

            history = history or []

            history.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": (
                        "Please upload a PDF document first."
                    ),
                }
            )

            return history

        try:

            response = self.chatbot.ask(
                message
            )

            answer = response.get(
                "answer",
                ""
            )

            history = history or []

            history.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            return history

        except Exception as exc:

            print()
            print(
                "ERROR WHILE ANSWERING:"
            )
            print(exc)

            history = history or []

            history.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Unable to answer the question: {exc}"
                    ),
                }
            )

            return history