import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    BGE-M3 embedding service.

    Responsible only for generating embeddings.

    Does NOT handle:
        - FAISS
        - BM25
        - RRF
        - Reranking
        - LLM generation
    """

    MODEL_NAME = "BAAI/bge-m3"

    BATCH_SIZE = 8

    NORMALIZE_EMBEDDINGS = True

    def __init__(
        self,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE
    ):

        self.model_name = model_name
        self.batch_size = batch_size

        print()
        print("Loading BGE-M3...")
        print(f"Model: {self.model_name}")

        self.model = SentenceTransformer(
            self.model_name
        )

        print("BGE-M3 loaded successfully.")

    # ========================================================
    # BUILD EMBEDDING TEXT
    # ========================================================

    def build_embedding_text(
        self,
        chunk
    ):
        """
        Build the text representation used for embedding.

        Includes useful structural information:
            - content type
            - section
            - caption
            - actual content
        """

        parts = []

        # ----------------------------------------------------
        # Content type
        # ----------------------------------------------------

        content_type = (
            chunk.get("content_type")
            or "text"
        )

        parts.append(
            f"Content Type: {content_type}"
        )

        # ----------------------------------------------------
        # Section
        # ----------------------------------------------------

        section_path = (
            chunk.get("section_path")
            or []
        )

        if section_path:

            parts.append(
                "Section: "
                + " > ".join(section_path)
            )

        # ----------------------------------------------------
        # Caption
        # ----------------------------------------------------

        caption = chunk.get(
            "caption"
        )

        if caption:

            parts.append(
                "Caption: "
                + caption
            )

        # ----------------------------------------------------
        # Main text
        # ----------------------------------------------------

        text = chunk.get(
            "text",
            ""
        )

        if text:

            parts.append(
                text
            )

        return "\n\n".join(parts)

    # ========================================================
    # DOCUMENT EMBEDDINGS
    # ========================================================

    def embed_documents(
        self,
        chunks
    ):
        """
        Generate embeddings for document chunks.

        Returns:
            numpy.ndarray

        Shape:
            (number_of_chunks, 1024)
        """

        if not chunks:

            return np.empty(
                (0, 0),
                dtype=np.float32
            )

        texts = [
            self.build_embedding_text(
                chunk
            )
            for chunk in chunks
        ]

        print()
        print("Creating document embeddings...")

        print(
            f"Chunks     : {len(texts)}"
        )

        print(
            f"Batch size : {self.batch_size}"
        )

        embeddings = self.model.encode(

            texts,

            batch_size=self.batch_size,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=(
                self.NORMALIZE_EMBEDDINGS
            )
        )

        embeddings = (
            embeddings.astype(
                np.float32
            )
        )

        print()
        print(
            "Embedding generation complete."
        )

        print(
            f"Shape: {embeddings.shape}"
        )

        return embeddings

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        query
    ):
        """
        Generate an embedding for a user query.

        Returns:
            numpy.ndarray

        Shape:
            (1, 1024)
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(

            [query],

            convert_to_numpy=True,

            normalize_embeddings=(
                self.NORMALIZE_EMBEDDINGS
            )
        )

        embedding = (
            embedding.astype(
                np.float32
            )
        )

        return embedding

    # ========================================================
    # VECTOR DIMENSION
    # ========================================================

    @property
    def dimension(self):

        return (
            self.model.get_sentence_embedding_dimension()
        )