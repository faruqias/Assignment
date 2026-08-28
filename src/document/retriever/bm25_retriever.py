import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 keyword-based retriever.

    Responsibilities:
        - Build BM25 index from chunks
        - Search chunks using a query
        - Return ranked vector IDs and BM25 scores

    Does NOT handle:
        - Embeddings
        - FAISS
        - RRF
        - Reranking
        - LLM generation
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        chunks,
        top_k=10
    ):

        self.chunks = chunks
        self.top_k = top_k

        print()
        print("Building BM25...")

        # ----------------------------------------------------
        # Create corpus
        # ----------------------------------------------------

        corpus = [
            self._tokenize(
                chunk.get(
                    "text",
                    ""
                )
            )
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            corpus
        )

        print(
            f"BM25 documents: {len(corpus)}"
        )

    # ========================================================
    # TOKENIZER
    # ========================================================

    @staticmethod
    def _tokenize(
        text
    ):
        """
        Simple tokenizer.

        We intentionally keep the same tokenizer
        used in the validated Step 5 implementation.
        """

        if not text:

            return []

        return text.lower().split()

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query
    ):
        """
        Search the BM25 index.

        Returns:

        [
            {
                "vector_id": 7,
                "rank": 1,
                "score": 12.34
            },
            ...
        ]
        """

        if not query or not query.strip():

            return []

        scores = self.bm25.get_scores(
            self._tokenize(query)
        )

        ranked_ids = np.argsort(
            scores
        )[::-1]

        results = []

        for rank, vector_id in enumerate(
            ranked_ids[:self.top_k],
            start=1
        ):

            results.append(
                {
                    "vector_id": int(
                        vector_id
                    ),

                    "rank": rank,

                    "score": float(
                        scores[vector_id]
                    )
                }
            )

        return results

    # ========================================================
    # SEARCH WITH METADATA
    # ========================================================

    def search_with_metadata(
        self,
        query
    ):
        """
        Search BM25 and attach chunk information.

        Useful for debugging and testing.
        """

        results = self.search(
            query
        )

        enriched_results = []

        for result in results:

            vector_id = result[
                "vector_id"
            ]

            chunk = self.chunks[
                vector_id
            ]

            enriched_results.append(
                {
                    **result,

                    "chunk_id": chunk.get(
                        "chunk_id"
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

                    "section_path": chunk.get(
                        "section_path",
                        []
                    )
                }
            )

        return enriched_results

    # ========================================================
    # DOCUMENT COUNT
    # ========================================================

    @property
    def document_count(self):

        return len(
            self.chunks
        )