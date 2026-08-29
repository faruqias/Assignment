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

from src.app.rag_chatbot import RAGChatbot


class RAGApplication:
    """
    Main RAG application orchestrator.

    Document ingestion pipeline:

        PDF
         ↓
        DocumentProcessor
         ↓
        DocumentParser
         ↓
        StructureChunker
         ↓
        EmbeddingService
         ↓
        VectorIndexer
         ↓
        BM25Retriever

    Query pipeline:

        Question
         ↓
        Retriever
         ├── Dense / FAISS
         └── BM25
              ↓
            RRF Fusion
              ↓
        BGE Reranker
              ↓
        RAGChatbot
              ↓
           Ollama
    """

    def __init__(self):

        print()
        print("=" * 70)
        print("INITIALIZING RAG APPLICATION")
        print("=" * 70)

        # ====================================================
        # APPLICATION STATE
        # ====================================================

        self.current_document_id = None
        self.current_document_name = None
        self.current_chunks = []

        self.bm25 = None
        self.rrf = None
        self.retriever = None

        # ====================================================
        # 1. DOCUMENT PROCESSOR
        # ====================================================

        print()
        print("1. Document Processor")

        self.processor = DocumentProcessor()

        # ====================================================
        # 2. DOCUMENT PARSER
        # ====================================================

        print()
        print("2. Document Parser")

        self.parser = DocumentParser()

        # ====================================================
        # 3. STRUCTURE CHUNKER
        # ====================================================

        print()
        print("3. Structure Chunker")

        self.chunker = StructureChunker()

        # ====================================================
        # 4. EMBEDDING SERVICE
        # ====================================================

        print()
        print("4. Embedding Service")

        self.embedding = EmbeddingService()

        # ====================================================
        # 5. VECTOR INDEXER
        # ====================================================

        print()
        print("5. Vector Indexer")

        base_dir = Path(
            __file__
        ).resolve().parents[2]

        vectorstore_dir = (
            base_dir
            / "data"
            / "vectorstore"
            / "attention"
        )

        vectorstore_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.faiss_path = (
            vectorstore_dir
            / "index.faiss"
        )

        self.metadata_path = (
            vectorstore_dir
            / "metadata.json"
        )

        self.indexer = VectorIndexer(
            index_path=self.faiss_path,
            metadata_path=self.metadata_path
        )

        # ====================================================
        # 6. RETRIEVER
        # ====================================================

        print()
        print("6. Retriever")

        # Retriever is initialized after document upload
        # because BM25Retriever requires chunks.

        # ====================================================
        # 7. BGE RERANKER
        # ====================================================

        print()
        print("7. BGE Reranker")

        self.reranker = BGEReranker()

        # ====================================================
        # 8. RAG CHATBOT
        # ====================================================

        print()
        print("8. RAG Chatbot")

        self.chatbot = RAGChatbot()

        print()
        print("=" * 70)
        print("RAG APPLICATION READY")
        print("=" * 70)

    # ========================================================
    # UPLOAD DOCUMENT
    # ========================================================

    def upload_document(
        self,
        pdf_path
    ):
        """
        Process, parse, chunk, embed and index a PDF.

        Returns:
            str: upload status message
        """

        if not pdf_path:

            return (
                "Please select a PDF document."
            )

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():

            return (
                f"PDF file not found: {pdf_path}"
            )

        print()
        print("=" * 70)
        print("DOCUMENT UPLOAD")
        print("=" * 70)

        print(
            f"PDF: {pdf_path}"
        )

        try:

            # =================================================
            # 1. DOCUMENT PROCESSOR
            # =================================================

            print()
            print(
                "1. Processing document..."
            )

            processor_result = (
                self.processor.process(
                    str(pdf_path)
                )
            )

            if not processor_result:

                return (
                    "Document processing returned no result."
                )

            document_id = (
                processor_result.get(
                    "document_id"
                )
            )

            document_name = (
                processor_result.get(
                    "document_name"
                )
            )

            json_path = (
                processor_result.get(
                    "json_path"
                )
            )

            print()
            print(
                f"Document ID   : {document_id}"
            )

            print(
                f"Document Name : {document_name}"
            )

            print(
                f"JSON          : {json_path}"
            )

            if not json_path:

                return (
                    "Document processor did not return "
                    "a JSON path."
                )

            # =================================================
            # 2. DOCUMENT PARSER
            # =================================================

            print()
            print(
                "2. Parsing document..."
            )

            # IMPORTANT:
            #
            # DocumentParser expects the Docling JSON path.
            #
            # Do NOT pass processor_result directly.
            #
            elements = self.parser.parse(
                json_path
            )

            print(
                f"   Parsed elements: {len(elements)}"
            )

            if not elements:

                return (
                    "No elements were extracted "
                    "from the document."
                )

            # =================================================
            # 3. STRUCTURE CHUNKER
            # =================================================

            print()
            print(
                "3. Building chunks..."
            )

            chunks = self.chunker.chunk(
                elements
            )

            print(
                f"   Chunks: {len(chunks)}"
            )

            if not chunks:

                return (
                    "No chunks were generated "
                    "from the document."
                )

            # =================================================
            # 4. ADD DOCUMENT METADATA
            # =================================================

            print()
            print(
                "4. Adding document metadata..."
            )

            for chunk in chunks:

                if not chunk.get(
                    "document_id"
                ):

                    chunk["document_id"] = (
                        document_id
                    )

                if not chunk.get(
                    "document_name"
                ):

                    chunk["document_name"] = (
                        document_name
                    )

            # =================================================
            # 5. EMBEDDINGS
            # =================================================

            print()
            print(
                "5. Generating embeddings..."
            )

            embeddings = (
                self.embedding.embed_documents(
                    chunks
                )
            )

            print(
                f"   Embedding shape: "
                f"{embeddings.shape}"
            )

            if len(embeddings) != len(chunks):

                raise RuntimeError(
                    "Embedding count does not match "
                    "chunk count."
                )

            # =================================================
            # 6. VECTOR INDEX
            # =================================================

            print()
            print(
                "6. Indexing vectors..."
            )

            self.indexer.index_documents(
                chunks,
                embeddings
            )

            self.indexer.save()

            print(
                f"   FAISS vectors: "
                f"{self.indexer.vector_count}"
            )

            # =================================================
            # 7. BM25
            # =================================================

            print()
            print(
                "7. Building BM25 index..."
            )

            self.bm25 = BM25Retriever(
                chunks
            )

            # =================================================
            # 8. RRF
            # =================================================

            print()
            print(
                "8. Initializing RRF fusion..."
            )

            self.rrf = RRFFusion()

            # =================================================
            # 9. RETRIEVER
            # =================================================

            print()
            print(
                "9. Initializing Retriever..."
            )

            self.retriever = Retriever(

                indexer=self.indexer,

                embedding_service=self.embedding,

                bm25_retriever=self.bm25,

                rrf_fusion=self.rrf
            )

            # =================================================
            # SAVE CURRENT DOCUMENT STATE
            # =================================================

            self.current_document_id = (
                document_id
            )

            self.current_document_name = (
                document_name
            )

            self.current_chunks = chunks

            print()
            print("=" * 70)
            print("DOCUMENT UPLOAD COMPLETED")
            print("=" * 70)

            print(
                f"Document : {document_name}"
            )

            print(
                f"Elements : {len(elements)}"
            )

            print(
                f"Chunks   : {len(chunks)}"
            )

            print(
                f"Vectors  : {len(embeddings)}"
            )

            return (
                f"Document uploaded successfully. "
                f"Chunks: {len(chunks)}"
            )

        except Exception as ex:

            print()
            print(
                "=" * 70
            )
            print(
                "ERROR WHILE PROCESSING DOCUMENT"
            )
            print(
                "=" * 70
            )

            print(
                f"{type(ex).__name__}: {ex}"
            )

            return (
                f"Error processing document: {ex}"
            )

    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask_question(
        self,
        question,
        history=None
    ):
        """
        Retrieve relevant chunks, rerank them and
        stream the generated answer.

        Yields:
            tuple[str, list]
        """

        if not question or not question.strip():

            yield (
                "Please enter a question.",
                []
            )

            return

        if self.retriever is None:

            yield (
                "Please upload a document before "
                "asking a question.",
                []
            )

            return

        question = question.strip()

        print()
        print("=" * 70)
        print("RAG QUERY")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        try:

            # =================================================
            # 1. RETRIEVAL
            # =================================================

            print()
            print(
                "1. Retrieving documents..."
            )

            retrieval_results = (
                self.retriever.retrieve(
                    question
                )
            )

            print(
                f"   Retrieved: "
                f"{len(retrieval_results)}"
            )

            if not retrieval_results:

                print(
                    "   No relevant documents found."
                )

                yield (
                    self.chatbot.FALLBACK_MESSAGE,
                    []
                )

                return

            # =================================================
            # 2. RERANKING
            # =================================================

            print()
            print(
                "2. Reranking..."
            )

            final_results = (
                self.reranker.rerank(
                    question,
                    retrieval_results,
                    self.current_chunks
                )
            )

            print(
                f"   Final results: "
                f"{len(final_results)}"
            )

            if not final_results:

                yield (
                    self.chatbot.FALLBACK_MESSAGE,
                    []
                )

                return

            # =================================================
            # 3. GENERATE ANSWER
            # =================================================

            print()
            print(
                "3. Generating answer..."
            )

            answer = ""

            for token in self.chatbot.stream(
                question,
                final_results
            ):

                answer += token

                yield (
                    answer,
                    final_results
                )

            print()
            print(
                "RAG query completed."
            )

        except Exception as ex:

            print()
            print(
                "=" * 70
            )
            print(
                "ERROR WHILE ANSWERING QUESTION"
            )
            print(
                "=" * 70
            )

            print(
                f"{type(ex).__name__}: {ex}"
            )

            yield (
                f"Error: {ex}",
                []
            )

    # ========================================================
    # SOURCES
    # ========================================================

    def get_sources(
        self,
        results
    ):
        """
        Convert retrieval results into displayable
        source information.
        """

        if not results:

            return []

        return self.chatbot.build_sources(
            results
        )