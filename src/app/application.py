from pathlib import Path
import json

import numpy as np
import faiss

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
from src.app.azure_openai_client import AzureOpenAIClient


class RAGApplication:
    """
    Main RAG application orchestrator.

    VECTOR STORE:

        data/vectorstore/
            index.faiss
            metadata.json

    All uploaded documents share the same vector index.

    Document pipeline:

        PDF
         ↓
        Docling
         ↓
        Parser
         ↓
        Chunker
         ↓
        BGE-M3
         ↓
        Unified FAISS
         ↓
        Unified metadata

    Query pipeline:

        Question
             ↓
        BGE-M3 query embedding
             ↓
        FAISS + BM25
             ↓
             RRF
             ↓
        BGE Reranker
             ↓
        Context selection
             ↓
        Prompt Builder
             ↓
        Azure OpenAI
    """

    # =========================================================
    # PATHS
    # =========================================================

    BASE_DIR = Path(
        __file__
    ).resolve().parents[2]

    VECTORSTORE_ROOT = (
        BASE_DIR
        / "data"
        / "vectorstore"
    )

    FAISS_PATH = (
        VECTORSTORE_ROOT
        / "index.faiss"
    )

    METADATA_PATH = (
        VECTORSTORE_ROOT
        / "metadata.json"
    )

    # =========================================================
    # INITIALIZATION
    # =========================================================

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
        # 5. UNIFIED VECTOR INDEX
        # =====================================================

        print()
        print("5. Unified Vector Index")

        self.VECTORSTORE_ROOT.mkdir(
            parents=True,
            exist_ok=True
        )

        self.indexer = VectorIndexer(
            index_path=self.FAISS_PATH,
            metadata_path=self.METADATA_PATH
        )

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
        # 8. AZURE OPENAI
        # =====================================================

        print()
        print("8. Azure OpenAI Client")

        self.openapi_client = AzureOpenAIClient()

        # =====================================================
        # RAG COMPONENTS
        # =====================================================

        self.bm25 = None
        self.rrf = None
        self.retriever = None
        self.chatbot = None

        # All chunks currently represented in the
        # unified vector store.
        self.chunks = []

        # =====================================================
        # LOAD EXISTING INDEX
        # =====================================================

        self._load_existing_index()

        print()
        print("=" * 70)
        print("RAG APPLICATION READY")
        print("=" * 70)

    # =========================================================
    # LOAD EXISTING UNIFIED INDEX
    # =========================================================

    def _load_existing_index(self):
        """
        Load the single unified FAISS index.

        No PDF processing.
        No document embedding.

        Existing vectors and metadata are reused.
        """

        if not (
            self.FAISS_PATH.exists()
            and self.METADATA_PATH.exists()
        ):

            print()
            print(
                "No existing unified vector store found."
            )

            return False

        print()
        print(
            "Loading existing unified vector store..."
        )

        self.indexer.load()

        self.chunks = list(
            self.indexer.metadata
        )

        if not self.chunks:

            print(
                "Existing vector store contains no metadata."
            )

            return False

        print(
            f"   Vectors: "
            f"{self.indexer.vector_count}"
        )

        print(
            f"   Chunks : "
            f"{len(self.chunks)}"
        )

        self._build_retrieval_pipeline()

        print(
            "Unified vector store loaded."
        )

        return True

    # =========================================================
    # BUILD RETRIEVAL PIPELINE
    # =========================================================

    def _build_retrieval_pipeline(self):
        """
        Build BM25, RRF, Retriever and RAGChatbot
        using the unified vector store.
        """

        if not self.chunks:

            self.bm25 = None
            self.rrf = None
            self.retriever = None
            self.chatbot = None

            return

        # -----------------------------------------------------
        # BM25
        # -----------------------------------------------------

        self.bm25 = BM25Retriever(
            self.chunks
        )

        # -----------------------------------------------------
        # RRF
        # -----------------------------------------------------

        self.rrf = RRFFusion()

        # -----------------------------------------------------
        # Retriever
        # -----------------------------------------------------

        self.retriever = Retriever(
            indexer=self.indexer,
            embedding_service=self.embedding,
            bm25_retriever=self.bm25,
            rrf_fusion=self.rrf
        )

        # -----------------------------------------------------
        # RAG Chatbot
        # -----------------------------------------------------

        self.chatbot = RAGChatbot(
            retriever=self.retriever,
            reranker=self.reranker,
            prompt_builder=self.prompt_builder,
            openapi_client=self.openapi_client,
            chunks=self.chunks,
        )

    # =========================================================
    # LOAD EMBEDDINGS FROM EXISTING FAISS INDEX
    # =========================================================

    def _read_existing_embeddings(self):
        """
        Read all existing vectors from the FAISS index.

        IndexFlatIP stores vectors directly.
        """

        if (
            self.indexer.index is None
            or self.indexer.index.ntotal == 0
        ):

            return None

        return self.indexer.index.reconstruct_n(
            0,
            self.indexer.index.ntotal
        )

    # =========================================================
    # REMOVE DUPLICATE CHUNKS
    # =========================================================

    @staticmethod
    def _chunk_key(chunk):
        """
        Create a stable key for duplicate detection.

        Document + chunk id is preferred.
        Text is used as a fallback.
        """

        document_id = chunk.get(
            "document_id",
            ""
        )

        chunk_id = chunk.get(
            "chunk_id",
            ""
        )

        text = chunk.get(
            "text",
            ""
        )

        if document_id and chunk_id:

            return (
                document_id,
                chunk_id
            )

        return (
            document_id,
            text.strip()
        )

    # =========================================================
    # UPLOAD DOCUMENTS
    # =========================================================

    def upload_documents(
        self,
        files
    ):
        """
        Process and add uploaded PDFs to the
        single unified vector store.

        IMPORTANT:

        This method never creates:

            vectorstore/<document_id>/

        All documents are stored in:

            data/vectorstore/index.faiss
            data/vectorstore/metadata.json
        """

        if not files:

            return (
                "Please upload at least one PDF document."
            )

        if not isinstance(
            files,
            list
        ):

            files = [files]

        messages = []

        # -----------------------------------------------------
        # Existing chunks
        # -----------------------------------------------------

        existing_keys = {
            self._chunk_key(chunk)
            for chunk in self.chunks
        }

        new_chunks = []

        # =====================================================
        # PROCESS UPLOADS
        # =====================================================

        for file_path in files:

            try:

                file_path = str(
                    file_path
                )

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

                # ------------------------------------------------
                # Check whether document already exists
                # ------------------------------------------------

                existing_document = any(
                    chunk.get(
                        "document_id"
                    ) == document_id
                    for chunk in self.chunks
                )

                if existing_document:

                    messages.append(
                        f"⚠ {document_name} "
                        "already indexed. Skipped."
                    )

                    continue

                # ------------------------------------------------
                # 1. PROCESS
                # ------------------------------------------------

                print()
                print(
                    "1. Processing document..."
                )

                result = (
                    self.processor.process(
                        file_path
                    )
                )

                # ------------------------------------------------
                # 2. PARSE
                # ------------------------------------------------

                print()
                print(
                    "2. Parsing document..."
                )

                elements = (
                    self.parser.parse(
                        result["json_path"]
                    )
                )

                # ------------------------------------------------
                # 3. CHUNK
                # ------------------------------------------------

                print()
                print(
                    "3. Building chunks..."
                )

                chunks = (
                    self.chunker.chunk(
                        elements
                    )
                )

                if not chunks:

                    messages.append(
                        f"✗ {document_name}: "
                        "no chunks generated."
                    )

                    continue

                # ------------------------------------------------
                # 4. DOCUMENT METADATA
                # ------------------------------------------------

                for chunk in chunks:

                    chunk[
                        "document_id"
                    ] = document_id

                    chunk[
                        "document_name"
                    ] = document_name

                # ------------------------------------------------
                # Duplicate chunk protection
                # ------------------------------------------------

                unique_chunks = []

                for chunk in chunks:

                    key = (
                        self._chunk_key(
                            chunk
                        )
                    )

                    if key in existing_keys:

                        continue

                    existing_keys.add(
                        key
                    )

                    unique_chunks.append(
                        chunk
                    )

                if not unique_chunks:

                    messages.append(
                        f"⚠ {document_name}: "
                        "all chunks already indexed."
                    )

                    continue

                new_chunks.extend(
                    unique_chunks
                )

                messages.append(
                    f"✓ {document_name}: "
                    f"{len(unique_chunks)} new chunks"
                )

            except Exception as exc:

                messages.append(
                    f"✗ {document_name}: "
                    f"{exc}"
                )

                print()
                print(
                    f"ERROR processing "
                    f"{document_name}:"
                )

                print(exc)

        # =====================================================
        # NOTHING NEW
        # =====================================================

        if not new_chunks:

            return "\n".join(
                messages
            )

        # =====================================================
        # EMBED ONLY NEW CHUNKS
        # =====================================================

        print()
        print(
            "=" * 60
        )
        print(
            "GENERATING EMBEDDINGS FOR NEW CHUNKS"
        )
        print(
            "=" * 60
        )

        new_embeddings = (
            self.embedding.embed_documents(
                new_chunks
            )
        )

        # =====================================================
        # EXISTING EMBEDDINGS
        # =====================================================

        existing_embeddings = (
            self._read_existing_embeddings()
        )

        # =====================================================
        # COMBINE
        # =====================================================

        if existing_embeddings is not None:

            combined_embeddings = (
                np.vstack(
                    [
                        existing_embeddings,
                        np.asarray(
                            new_embeddings,
                            dtype=np.float32
                        )
                    ]
                )
            )

            combined_chunks = (
                self.chunks
                + new_chunks
            )

        else:

            combined_embeddings = (
                np.asarray(
                    new_embeddings,
                    dtype=np.float32
                )
            )

            combined_chunks = (
                new_chunks
            )

        # =====================================================
        # REBUILD SINGLE FAISS INDEX
        # =====================================================

        print()
        print(
            "=" * 60
        )
        print(
            "UPDATING UNIFIED VECTOR STORE"
        )
        print(
            "=" * 60
        )

        self.indexer.index_documents(
            combined_chunks,
            combined_embeddings
        )

        self.indexer.save()

        # =====================================================
        # UPDATE APPLICATION STATE
        # =====================================================

        self.chunks = combined_chunks

        self._build_retrieval_pipeline()

        print()
        print(
            f"Unified vectors: "
            f"{len(self.chunks)}"
        )

        return "\n".join(
            messages
        )

    # =========================================================
    # ASK QUESTION
    # =========================================================

    def ask_question(
        self,
        message,
        history
    ):
        """
        Ask a question using the unified vector store.

        No upload is required for every session.
        """

        if not message or not message.strip():

            return history or []

        # -----------------------------------------------------
        # If application started before an index existed,
        # try loading it now.
        # -----------------------------------------------------

        if self.chatbot is None:

            try:

                self._load_existing_index()

            except Exception as exc:

                print()
                print(
                    "ERROR LOADING VECTOR STORE:"
                )

                print(exc)

        # -----------------------------------------------------
        # No index
        # -----------------------------------------------------

        if self.chatbot is None:

            history = (
                history or []
            )

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
                        "I couldn't find any indexed "
                        "document to answer this question."
                    ),
                }
            )

            return history

        # -----------------------------------------------------
        # Ask RAG
        # -----------------------------------------------------

        try:

            response = (
                self.chatbot.ask(
                    message
                )
            )

            answer = response.get(
                "answer",
                ""
            )

            history = (
                history or []
            )

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

            history = (
                history or []
            )

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
                        f"Unable to answer "
                        f"the question: {exc}"
                    ),
                }
            )

            return history