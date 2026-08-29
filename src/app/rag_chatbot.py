import time
import os

from dotenv import load_dotenv
import ollama

from src.app.prompt_builder import PromptBuilder


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

OLLAMA_TEMPERATURE = float(
    os.getenv(
        "OLLAMA_TEMPERATURE",
        "0.2"
    )
)

OLLAMA_MAX_TOKENS = int(
    os.getenv(
        "OLLAMA_MAX_TOKENS",
        "300"
    )
)

MAX_CONTEXT_RESULTS = int(
    os.getenv(
        "MAX_CONTEXT_RESULTS",
        "3"
    )
)

MAX_CHUNK_CHARS = int(
    os.getenv(
        "MAX_CHUNK_CHARS",
        "3000"
    )
)


class RAGChatbot:
    """
    RAG generation layer.

    Responsibilities:
        1. Build compact context.
        2. Build document-grounded prompt.
        3. Send prompt to Ollama.
        4. Stream generated answer.
        5. Build source information.

    Does NOT perform:
        - PDF processing
        - Parsing
        - Chunking
        - Embedding
        - FAISS
        - BM25
        - RRF
        - Retrieval
        - Reranking
    """

    FALLBACK_MESSAGE = (
        "I couldn't find this information "
        "in the uploaded document."
    )

    def __init__(
        self,
        model=OLLAMA_MODEL
    ):

        self.model = model

        self.prompt_builder = PromptBuilder()

        print()
        print("RAG Chatbot")
        print(
            f"LLM Model: {self.model}"
        )

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(
        self,
        results
    ):
        """
        Build compact LLM context.

        Only the most relevant reranked results are
        passed to the LLM.

        Retrieval/reranking metadata is intentionally
        kept minimal to reduce prompt size.
        """

        if not results:
            return ""

        context_parts = []

        # ----------------------------------------------------
        # Only use top reranked results
        # ----------------------------------------------------

        selected_results = results[
            :MAX_CONTEXT_RESULTS
        ]

        for rank, result in enumerate(
            selected_results,
            start=1
        ):

            text = result.get(
                "text",
                ""
            )

            if not text:

                text = result.get(
                    "page_content",
                    ""
                )

            text = str(
                text
            ).strip()

            if not text:
                continue

            # ------------------------------------------------
            # Limit context size
            # ------------------------------------------------

            if len(text) > MAX_CHUNK_CHARS:

                text = (
                    text[:MAX_CHUNK_CHARS]
                    + "\n[Content truncated]"
                )

            # ------------------------------------------------
            # Section
            # ------------------------------------------------

            section_path = result.get(
                "section_path",
                []
            )

            section = ""

            if section_path:

                section = (
                    "\nSection: "
                    +
                    " > ".join(
                        section_path
                    )
                )

            # ------------------------------------------------
            # Build compact context
            # ------------------------------------------------

            context_parts.append(
                f"[Source {rank}]"
                f"{section}"
                f"\n{text}"
            )

        return "\n\n".join(
            context_parts
        )

    # ========================================================
    # STREAM
    # ========================================================

    def stream(
        self,
        question,
        results
    ):
        """
        Generate and stream a document-grounded answer.
        """

        if not question or not question.strip():

            yield self.FALLBACK_MESSAGE

            return

        if not results:

            yield self.FALLBACK_MESSAGE

            return

        # ----------------------------------------------------
        # Build compact context
        # ----------------------------------------------------

        context = self.build_context(
            results
        )

        if not context.strip():

            yield self.FALLBACK_MESSAGE

            return

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = self.prompt_builder.build_prompt(
            question=question,
            context=context
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print()
        print(
            "Sending prompt to Ollama..."
        )

        print(
            f"Model      : {self.model}"
        )

        print(
            f"Results    : "
            f"{min(len(results), MAX_CONTEXT_RESULTS)}"
        )

        print(
            f"Context    : "
            f"{len(context):,} characters"
        )

        print(
            f"Prompt     : "
            f"{len(prompt):,} characters"
        )

        start_time = time.perf_counter()

        first_token_time = None

        # ----------------------------------------------------
        # Ollama
        # ----------------------------------------------------

        stream = ollama.chat(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            stream=True,

            options={
                "temperature":
                    OLLAMA_TEMPERATURE,

                "num_predict":
                    OLLAMA_MAX_TOKENS
            }
        )

        # ----------------------------------------------------
        # Stream response
        # ----------------------------------------------------

        for response in stream:

            token = (
                response
                .get(
                    "message",
                    {}
                )
                .get(
                    "content",
                    ""
                )
            )

            if not token:
                continue

            if first_token_time is None:

                first_token_time = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"First token: "
                    f"{first_token_time:.3f} sec"
                )

            yield token

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Ollama total: "
            f"{total_time:.3f} sec"
        )

    # ========================================================
    # COMPLETE ANSWER
    # ========================================================

    def answer(
        self,
        question,
        results
    ):
        """
        Generate a complete answer.
        """

        answer = ""

        for token in self.stream(
            question,
            results
        ):

            answer += token

        return answer.strip()

    # ========================================================
    # BUILD SOURCES
    # ========================================================

    def build_sources(
        self,
        results
    ):
        """
        Build readable source information.

        Source generation is independent from the
        compact LLM context.
        """

        if not results:
            return ""

        sources = []

        seen = set()

        for result in results:

            chunk_id = result.get(
                "chunk_id",
                "Unknown"
            )

            page_start = result.get(
                "page_start"
            )

            page_end = result.get(
                "page_end"
            )

            section_path = result.get(
                "section_path",
                []
            )

            # ------------------------------------------------
            # Prevent duplicate sources
            # ------------------------------------------------

            key = (
                chunk_id,
                page_start,
                page_end
            )

            if key in seen:
                continue

            seen.add(key)

            # ------------------------------------------------
            # Source
            # ------------------------------------------------

            source = (
                f"• {chunk_id}"
            )

            if page_start is not None:

                source += (
                    f" | Page {page_start}"
                )

                if (
                    page_end is not None
                    and page_end != page_start
                ):

                    source += (
                        f"-{page_end}"
                    )

            if section_path:

                source += (
                    " | "
                    +
                    " > ".join(
                        section_path
                    )
                )

            sources.append(
                source
            )

        return "\n".join(
            sources
        )