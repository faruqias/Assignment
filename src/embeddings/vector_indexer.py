from pathlib import Path
import json

import faiss
import numpy as np


class VectorIndexer:
    """
    FAISS vector index manager.

    Responsibilities:
        - Create FAISS index
        - Add document embeddings
        - Search vectors
        - Save index
        - Save metadata
        - Load existing index

    Does NOT handle:
        - PDF processing
        - Chunking
        - Embedding generation
        - BM25
        - RRF
        - Reranking
        - LLM generation
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        index_path=None,
        metadata_path=None
    ):

        self.index_path = (
            Path(index_path)
            if index_path
            else None
        )

        self.metadata_path = (
            Path(metadata_path)
            if metadata_path
            else None
        )

        self.index = None

        self.metadata = []

    # ========================================================
    # CREATE INDEX
    # ========================================================

    def create_index(
        self,
        embeddings
    ):
        """
        Create a FAISS Inner Product index.

        Embeddings must already be normalized.

        For normalized vectors:

            Inner Product ≈ Cosine Similarity
        """

        if embeddings is None:

            raise ValueError(
                "Embeddings cannot be None."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[0] == 0:

            raise ValueError(
                "No embeddings provided."
            )

        dimension = embeddings.shape[1]

        print()
        print(
            "Creating FAISS index..."
        )

        print(
            f"Vector dimension : {dimension}"
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        print(
            f"Vectors indexed  : "
            f"{self.index.ntotal}"
        )

        return self.index

    # ========================================================
    # INDEX DOCUMENTS
    # ========================================================

    def index_documents(
        self,
        chunks,
        embeddings
    ):
        """
        Create FAISS index and build metadata
        from document chunks.
        """

        if len(chunks) != len(embeddings):

            raise ValueError(
                "Number of chunks does not match "
                "number of embeddings."
            )

        self.create_index(
            embeddings
        )

        self.metadata = (
            self._build_metadata(
                chunks
            )
        )

        return self.index

    # ========================================================
    # BUILD METADATA
    # ========================================================

    def _build_metadata(
        self,
        chunks
    ):
        """
        Build metadata using the same vector position
        as the FAISS vector ID.

        vector_id 0 → chunks[0]
        vector_id 1 → chunks[1]
        ...
        """

        metadata = []

        for vector_id, chunk in enumerate(
            chunks
        ):

            metadata.append(
                {
                    "vector_id": vector_id,

                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),

                    "document_id": chunk.get(
                        "document_id"
                    ),

                    "document_name": chunk.get(
                        "document_name"
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

                    "section": chunk.get(
                        "section"
                    ),

                    "section_path": chunk.get(
                        "section_path",
                        []
                    ),

                    "token_count": chunk.get(
                        "token_count"
                    ),

                    "parent_id": chunk.get(
                        "parent_id"
                    ),

                    "is_atomic": chunk.get(
                        "is_atomic",
                        False
                    ),

                    "caption": chunk.get(
                        "caption"
                    ),

                    "image_path": chunk.get(
                        "image_path"
                    ),

                    "document_part": chunk.get(
                        "document_part",
                        "main"
                    ),

                    "referenced_from": chunk.get(
                        "referenced_from",
                        []
                    ),

                    # Keep original text available
                    # for retrieval / RAG context.
                    "text": chunk.get(
                        "text",
                        ""
                    )
                }
            )

        return metadata

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_vector,
        top_k=10
    ):
        """
        Search the FAISS index.

        query_vector should be a normalized
        embedding generated by EmbeddingService.
        """

        if self.index is None:

            raise RuntimeError(
                "FAISS index has not been created "
                "or loaded."
            )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Ensure shape is (1, dimension)
        # ----------------------------------------------------

        if query_vector.ndim == 1:

            query_vector = (
                query_vector.reshape(
                    1,
                    -1
                )
            )

        if query_vector.ndim != 2:

            raise ValueError(
                "Query vector must be a 1D or 2D array."
            )

        # ----------------------------------------------------
        # Do not request more vectors than exist
        # ----------------------------------------------------

        top_k = min(
            top_k,
            self.index.ntotal
        )

        scores, ids = self.index.search(
            query_vector,
            top_k
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

            result = {
                "vector_id": int(
                    vector_id
                ),

                "rank": rank,

                "score": float(
                    score
                )
            }

            # ------------------------------------------------
            # Attach metadata
            # ------------------------------------------------

            if (
                vector_id
                < len(self.metadata)
            ):

                result["metadata"] = (
                    self.metadata[
                        vector_id
                    ]
                )

            results.append(
                result
            )

        return results

    # ========================================================
    # SAVE INDEX
    # ========================================================

    def save_index(
        self,
        path=None
    ):
        """
        Save FAISS index to disk.
        """

        path = (
            Path(path)
            if path
            else self.index_path
        )

        if path is None:

            raise ValueError(
                "Index path is required."
            )

        if self.index is None:

            raise RuntimeError(
                "No FAISS index to save."
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(path)
        )

        self.index_path = path

        print(
            f"FAISS index saved: {path}"
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    def save_metadata(
        self,
        path=None
    ):
        """
        Save vector metadata to JSON.
        """

        path = (
            Path(path)
            if path
            else self.metadata_path
        )

        if path is None:

            raise ValueError(
                "Metadata path is required."
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        self.metadata_path = path

        print(
            f"Metadata saved: {path}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        index_path=None,
        metadata_path=None
    ):
        """
        Save both FAISS index and metadata.
        """

        self.save_index(
            index_path
        )

        self.save_metadata(
            metadata_path
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        index_path=None,
        metadata_path=None
    ):
        """
        Load an existing FAISS index and metadata.
        """

        index_path = (
            Path(index_path)
            if index_path
            else self.index_path
        )

        metadata_path = (
            Path(metadata_path)
            if metadata_path
            else self.metadata_path
        )

        if index_path is None:

            raise ValueError(
                "Index path is required."
            )

        if metadata_path is None:

            raise ValueError(
                "Metadata path is required."
            )

        if not index_path.exists():

            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{index_path}"
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata not found: "
                f"{metadata_path}"
            )

        print()
        print(
            "Loading FAISS index..."
        )

        self.index = faiss.read_index(
            str(index_path)
        )

        print(
            f"FAISS vectors: "
            f"{self.index.ntotal}"
        )

        print(
            "Loading metadata..."
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(
                f
            )

        print(
            f"Metadata records: "
            f"{len(self.metadata)}"
        )

        self.index_path = index_path

        self.metadata_path = metadata_path

        self._validate_loaded_data()

        return self

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_loaded_data(
        self
    ):

        if self.index is None:

            raise RuntimeError(
                "FAISS index is not loaded."
            )

        if (
            self.index.ntotal
            != len(self.metadata)
        ):

            raise RuntimeError(
                "FAISS vector count "
                "does not match metadata count."
            )

        print(
            "FAISS and metadata validation passed."
        )

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def vector_count(self):

        if self.index is None:

            return 0

        return self.index.ntotal

    @property
    def dimension(self):

        if self.index is None:

            return 0

        return self.index.d

    @property
    def ready(self):

        return (
            self.index is not None
            and self.index.ntotal > 0
        )