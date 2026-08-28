class RRFFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Combines ranked results from multiple retrieval
    methods such as:

        FAISS / Dense Retrieval
                 +
              BM25
                 ↓
                RRF
                 ↓
          Combined ranking

    Does NOT handle:
        - Embeddings
        - FAISS search
        - BM25 search
        - Reranking
        - LLM generation
    """

    def __init__(
        self,
        top_k=10,
        rrf_k=60
    ):

        self.top_k = top_k
        self.rrf_k = rrf_k

    # ========================================================
    # FUSE
    # ========================================================

    def fuse(
        self,
        dense_results,
        bm25_results
    ):
        """
        Combine dense and BM25 ranked results.

        Parameters
        ----------
        dense_results:
            Results from VectorIndexer.search()

        bm25_results:
            Results from BM25Retriever.search()

        Returns
        -------
        list[dict]
        """

        fused = {}

        dense_ranks = {}

        bm25_ranks = {}

        # ----------------------------------------------------
        # Dense results
        # ----------------------------------------------------

        for result in dense_results:

            vector_id = result[
                "vector_id"
            ]

            rank = result[
                "rank"
            ]

            dense_ranks[
                vector_id
            ] = rank

            fused[
                vector_id
            ] = (
                fused.get(
                    vector_id,
                    0.0
                )
                +
                1.0 / (
                    self.rrf_k + rank
                )
            )

        # ----------------------------------------------------
        # BM25 results
        # ----------------------------------------------------

        for result in bm25_results:

            vector_id = result[
                "vector_id"
            ]

            rank = result[
                "rank"
            ]

            bm25_ranks[
                vector_id
            ] = rank

            fused[
                vector_id
            ] = (
                fused.get(
                    vector_id,
                    0.0
                )
                +
                1.0 / (
                    self.rrf_k + rank
                )
            )

        # ----------------------------------------------------
        # Sort by RRF score
        # ----------------------------------------------------

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        results = []

        for vector_id, rrf_score in ranked[
            :self.top_k
        ]:

            results.append(
                {
                    "vector_id": int(
                        vector_id
                    ),

                    "rrf_score": float(
                        rrf_score
                    ),

                    "dense_rank":
                        dense_ranks.get(
                            vector_id
                        ),

                    "bm25_rank":
                        bm25_ranks.get(
                            vector_id
                        )
                }
            )

        return results