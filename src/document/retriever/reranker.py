import os

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

USE_RERANKER = (
    os.getenv(
        "USE_RERANKER",
        "true"
    ).lower()
    == "true"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3"
)

RERANKER_FP16 = (
    os.getenv(
        "RERANKER_FP16",
        "false"
    ).lower()
    == "true"
)

FINAL_TOP_K = int(
    os.getenv(
        "FINAL_TOP_K",
        "5"
    )
)


# ============================================================
# BGE RERANKER
# ============================================================

class BGEReranker:
    """
    BGE Reranker service.

    Reranking can be enabled/disabled through .env:

        USE_RERANKER=true

    or:

        USE_RERANKER=false

    When disabled, RRF results are returned directly.
    """

    def __init__(
        self,
        model_name=RERANKER_MODEL,
        top_k=FINAL_TOP_K
    ):

        self.enabled = USE_RERANKER

        self.model_name = model_name

        self.top_k = top_k

        self.reranker = None

        # ----------------------------------------------------
        # Configuration information
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("BGE RERANKER")
        print("=" * 60)

        print(
            f"Enabled : {self.enabled}"
        )

        print(
            f"Model   : {self.model_name}"
        )

        print(
            f"FP16    : {RERANKER_FP16}"
        )

        print(
            f"Top K   : {self.top_k}"
        )

        # ----------------------------------------------------
        # Disabled
        # ----------------------------------------------------

        if not self.enabled:

            print()
            print(
                "BGE Reranker: DISABLED"
            )

            print("=" * 60)

            return

        # ----------------------------------------------------
        # Enabled
        # ----------------------------------------------------

        print()
        print(
            "Loading BGE Reranker..."
        )

        self.reranker = FlagReranker(
            self.model_name,
            use_fp16=RERANKER_FP16
        )

        print(
            "BGE Reranker: Ready"
        )

        print("=" * 60)

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query,
        candidates,
        chunks
    ):
        """
        Rerank RRF candidates.

        Parameters
        ----------
        query:
            User question.

        candidates:
            RRF candidate results.

        chunks:
            Loaded chunk collection.

        Returns
        -------
        list:
            Final ranked results.
        """

        if not candidates:

            return []

        # ----------------------------------------------------
        # Disabled
        # ----------------------------------------------------

        if not self.enabled:

            return self._without_reranking(
                candidates,
                chunks
            )

        # ----------------------------------------------------
        # Build query-document pairs
        # ----------------------------------------------------

        pairs = []

        for candidate in candidates:

            vector_id = candidate[
                "vector_id"
            ]

            chunk = chunks[
                vector_id
            ]

            candidate_text = (
                self._build_candidate_text(
                    chunk
                )
            )

            pairs.append(
                [
                    query,
                    candidate_text
                ]
            )

        # ----------------------------------------------------
        # Run BGE Reranker
        # ----------------------------------------------------

        scores = (
            self.reranker.compute_score(
                pairs,
                normalize=True
            )
        )

        # FlagEmbedding can return a
        # single number for one pair.

        if isinstance(
            scores,
            (float, int)
        ):

            scores = [scores]

        # ----------------------------------------------------
        # Build results
        # ----------------------------------------------------

        results = []

        for candidate, score in zip(
            candidates,
            scores
        ):

            vector_id = candidate[
                "vector_id"
            ]

            chunk = chunks[
                vector_id
            ]

            result = (
                self._build_result(
                    candidate=candidate,
                    chunk=chunk,
                    reranker_score=float(
                        score
                    )
                )
            )

            results.append(
                result
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        results.sort(
            key=lambda x:
                x["reranker_score"],
            reverse=True
        )

        return results[
            :self.top_k
        ]

    # ========================================================
    # WITHOUT RERANKING
    # ========================================================

    def _without_reranking(
        self,
        candidates,
        chunks
    ):
        """
        Return RRF results without reranking.

        RRF ordering is preserved.
        """

        results = []

        for candidate in candidates[
            :self.top_k
        ]:

            vector_id = candidate[
                "vector_id"
            ]

            chunk = chunks[
                vector_id
            ]

            result = (
                self._build_result(
                    candidate=candidate,
                    chunk=chunk,
                    reranker_score=None
                )
            )

            results.append(
                result
            )

        return results

    # ========================================================
    # BUILD CANDIDATE TEXT
    # ========================================================

    @staticmethod
    def _build_candidate_text(
        chunk
    ):
        """
        Build text sent to the reranker.

        Section + caption + content.
        """

        parts = []

        section_path = chunk.get(
            "section_path",
            []
        )

        caption = chunk.get(
            "caption"
        )

        text = chunk.get(
            "text",
            ""
        )

        if section_path:

            parts.append(
                "Section: "
                +
                " > ".join(
                    section_path
                )
            )

        if caption:

            parts.append(
                "Caption: "
                + caption
            )

        if text:

            parts.append(
                text
            )

        return "\n".join(
            parts
        )

    # ========================================================
    # BUILD RESULT
    # ========================================================

    @staticmethod
    def _build_result(
        candidate,
        chunk,
        reranker_score
    ):
        """
        Create the standard retrieval result
        used by the rest of the RAG pipeline.
        """

        return {

            "vector_id":
                candidate[
                    "vector_id"
                ],

            "chunk_id":
                chunk.get(
                    "chunk_id"
                ),

            "content_type":
                chunk.get(
                    "content_type"
                ),

            "page_start":
                chunk.get(
                    "page_start"
                ),

            "page_end":
                chunk.get(
                    "page_end"
                ),

            "section_path":
                chunk.get(
                    "section_path",
                    []
                ),

            "caption":
                chunk.get(
                    "caption"
                ),

            "text":
                chunk.get(
                    "text",
                    ""
                ),

            "dense_rank":
                candidate.get(
                    "dense_rank"
                ),

            "bm25_rank":
                candidate.get(
                    "bm25_rank"
                ),

            "rrf_score":
                candidate.get(
                    "rrf_score"
                ),

            "reranker_score":
                reranker_score
        }