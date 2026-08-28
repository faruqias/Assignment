import time


class Retriever:
    """
    Hybrid retrieval orchestrator.

    Pipeline:

        Query
          │
          ├── BGE-M3 → FAISS
          │
          └── BM25
                │
                ▼
             RRF Fusion
                │
                ▼
          RRF candidates

    Reranking is intentionally kept outside this class.
    """

    def __init__(
        self,
        indexer,
        embedding_service,
        bm25_retriever,
        rrf_fusion,
        dense_top_k=10,
        bm25_top_k=10,
        rrf_top_k=10
    ):

        self.indexer = indexer

        self.embedding_service = (
            embedding_service
        )

        self.bm25_retriever = (
            bm25_retriever
        )

        self.rrf_fusion = (
            rrf_fusion
        )

        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_top_k = rrf_top_k

    # ========================================================
    # DENSE RETRIEVAL
    # ========================================================

    def dense_search(
        self,
        query
    ):
        """
        Generate query embedding and search FAISS.
        """

        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        return self.indexer.search(
            query_vector,
            top_k=self.dense_top_k
        )

    # ========================================================
    # BM25 RETRIEVAL
    # ========================================================

    def keyword_search(
        self,
        query
    ):
        """
        Search using BM25.
        """

        return self.bm25_retriever.search(
            query
        )

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query
    ):
        """
        Execute:

            FAISS
              +
            BM25
              ↓
            RRF

        Returns RRF candidates.

        Reranking is handled separately.
        """

        if not query or not query.strip():

            return []

        total_start = time.perf_counter()

        print()
        print("=" * 60)
        print("HYBRID RETRIEVAL")
        print("=" * 60)

        print(
            f"Query: {query}"
        )

        # ----------------------------------------------------
        # Dense / FAISS
        # ----------------------------------------------------

        start = time.perf_counter()

        dense_results = (
            self.dense_search(
                query
            )
        )

        dense_time = (
            time.perf_counter()
            - start
        )

        print()
        print(
            f"1. FAISS / Dense : "
            f"{dense_time:.3f} sec"
        )

        print(
            f"   Results        : "
            f"{len(dense_results)}"
        )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        start = time.perf_counter()

        bm25_results = (
            self.keyword_search(
                query
            )
        )

        bm25_time = (
            time.perf_counter()
            - start
        )

        print()
        print(
            f"2. BM25          : "
            f"{bm25_time:.3f} sec"
        )

        print(
            f"   Results        : "
            f"{len(bm25_results)}"
        )

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        start = time.perf_counter()

        rrf_results = (
            self.rrf_fusion.fuse(
                dense_results,
                bm25_results
            )
        )

        rrf_time = (
            time.perf_counter()
            - start
        )

        print()
        print(
            f"3. RRF            : "
            f"{rrf_time:.3f} sec"
        )

        print(
            f"   Candidates      : "
            f"{len(rrf_results)}"
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        )

        print()
        print(
            f"Retrieval total   : "
            f"{total_time:.3f} sec"
        )

        print("=" * 60)

        return rrf_results

    # ========================================================
    # GET CHUNKS
    # ========================================================

    def get_chunks(
        self,
        results
    ):
        """
        Convert vector IDs from retrieval results
        into the corresponding chunks.

        Useful before reranking or generation.
        """

        chunks = []

        for result in results:

            vector_id = result[
                "vector_id"
            ]

            if (
                vector_id < 0
                or vector_id >= len(
                    self.bm25_retriever.chunks
                )
            ):

                continue

            chunks.append(
                self.bm25_retriever.chunks[
                    vector_id
                ]
            )

        return chunks