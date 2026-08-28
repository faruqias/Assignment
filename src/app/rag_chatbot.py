import time
import os
from dotenv import load_dotenv 
import ollama

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS"))


class RAGChatbot:
    """
    RAG generation layer.

    Responsibilities:
        1. Build context from retrieved chunks.
        2. Build a strict document-grounded prompt.
        3. Send prompt to Ollama.
        4. Stream the generated answer.
        5. Build source information.

    This class does NOT perform:
        - PDF processing
        - Parsing
        - Chunking
        - Embedding
        - FAISS search
        - BM25 search
        - RRF
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

        print()
        print("RAG Chatbot")
        print(
            f"LLM Model: {self.model}"
        )

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(self, results):
        """
        Convert retrieved results into the context
        supplied to the LLM.
        """

        if not results:
            return ""

        context_parts = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            parts = []

            parts.append(
                f"[Source {rank}]"
            )

            # ------------------------------------------------
            # Content type
            # ------------------------------------------------

            content_type = result.get(
                "content_type"
            )

            if content_type:

                parts.append(
                    f"Type: {content_type}"
                )

            # ------------------------------------------------
            # Page
            # ------------------------------------------------

            page_start = result.get(
                "page_start"
            )

            page_end = result.get(
                "page_end"
            )

            if page_start is not None:

                if (
                    page_end is not None
                    and page_end != page_start
                ):

                    parts.append(
                        f"Page: "
                        f"{page_start}-{page_end}"
                    )

                else:

                    parts.append(
                        f"Page: "
                        f"{page_start}"
                    )

            # ------------------------------------------------
            # Section
            # ------------------------------------------------

            section_path = result.get(
                "section_path",
                []
            )

            if section_path:

                parts.append(
                    "Section: "
                    +
                    " > ".join(
                        section_path
                    )
                )

            # ------------------------------------------------
            # Caption
            # ------------------------------------------------

            caption = result.get(
                "caption"
            )

            if caption:

                parts.append(
                    "Caption: "
                    + caption
                )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            text = result.get(
                "text",
                ""
            )

            if text:

                parts.append(
                    "Content:\n"
                    + text
                )

            context_parts.append(
                "\n".join(parts)
            )

        return "\n\n".join(
            context_parts
        )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def build_prompt(
        self,
        question,
        context
    ):
        """
        Build a strict document-grounded prompt.

        The model must answer only from the supplied
        document context.
        """

        return f"""
You are a document question-answering assistant.

Your task is to answer the QUESTION using ONLY the
DOCUMENT CONTEXT provided below.

STRICT RULES:

1. Use only information contained in the DOCUMENT CONTEXT.

2. Do not use your own knowledge.

3. Do not make assumptions or fill in missing information.

4. Every factual statement must be supported by the
   DOCUMENT CONTEXT.

5. Do not invent formulas, terminology, examples,
   numbers, names, or explanations.

6. If the context does not contain enough information
   to answer the question, respond exactly:

"I couldn't find this information in the uploaded document."

7. If a formula appears in the document context,
   reproduce it faithfully.

8. Do not create or modify a formula based on your
   own knowledge.

9. Start directly with the answer.

10. Do not say:
    - "Here is the answer..."
    - "Based on my knowledge..."
    - "According to my knowledge..."
    - "As an AI..."
    - "Based on the context..."

11. Keep the answer concise and clear.

12. Use bullet points when appropriate.

13. Mention the relevant page or section when the
    information is available.

14. Do not mention the retrieval system, FAISS, BM25,
    RRF, embeddings, reranking, or the RAG pipeline.

QUESTION:

{question}

DOCUMENT CONTEXT:

{context}

ANSWER:
""".strip()

    # ========================================================
    # STREAM
    # ========================================================

    def stream(
        self,
        question,
        results
    ):
        """
        Stream the answer from Ollama.

        IMPORTANT:
        This method yields ONLY the newly generated
        token/chunk.

        The caller is responsible for accumulating
        the tokens.
        """

        if not results:

            yield self.FALLBACK_MESSAGE

            return

        # ----------------------------------------------------
        # Build context
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

        prompt = self.build_prompt(
            question=question,
            context=context
        )

        # ----------------------------------------------------
        # Ollama
        # ----------------------------------------------------

        print()
        print(
            "Sending prompt to Ollama..."
        )

        print(
            f"Model      : {self.model}"
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
        # Stream tokens
        # ----------------------------------------------------

        for response in stream:

            token = (
                response
                .get("message", {})
                .get("content", "")
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

            # IMPORTANT:
            # Return ONLY the new token.
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
        Generate a complete non-streaming answer.
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