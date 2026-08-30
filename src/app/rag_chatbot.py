import os
import time
from typing import Any, Dict, List, Tuple, Iterator

import ollama


class RAGChatbot:
    """
    Retrieval-Augmented Generation chatbot.

    Pipeline:

        Question
            ↓
        Retriever
            ↓
        BGE Reranker
            ↓
        Context Limiting
            ↓
        PromptBuilder
            ↓
        Ollama
            ↓
        Answer + Sources

    Responsibilities:
        - Retrieve relevant documents
        - Rerank documents
        - Select final context
        - Build prompt
        - Generate answer
        - Stream answer
        - Track exact source chunks used

    Does NOT handle:
        - PDF processing
        - Parsing
        - Chunking
        - Embedding generation
        - FAISS
        - BM25
        - RRF
    """

    DEFAULT_MODEL = "llama3.2:latest"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 200
    DEFAULT_MAX_CONTEXT_RESULTS = 3
    DEFAULT_MAX_CONTEXT_CHARS = 5000

    FALLBACK_MESSAGE = (
        "I couldn't find this information "
        "in the uploaded document."
    )

    def __init__(
        self,
        retriever,
        reranker,
        prompt_builder,
        llm_client,
        chunks,
        model=None,
        temperature=None,
        max_tokens=None,
        max_context_results=None,
        max_context_chars=None,
    ):
        # -----------------------------------------------------
        # Dependencies
        # -----------------------------------------------------

        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.chunks = chunks

        # -----------------------------------------------------
        # Configuration
        # -----------------------------------------------------

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                self.DEFAULT_MODEL,
            )
        )

        self.temperature = (
            temperature
            if temperature is not None
            else float(
                os.getenv(
                    "OLLAMA_TEMPERATURE",
                    self.DEFAULT_TEMPERATURE,
                )
            )
        )

        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(
                os.getenv(
                    "OLLAMA_MAX_TOKENS",
                    self.DEFAULT_MAX_TOKENS,
                )
            )
        )

        self.max_context_results = (
            max_context_results
            if max_context_results is not None
            else int(
                os.getenv(
                    "MAX_CONTEXT_RESULTS",
                    self.DEFAULT_MAX_CONTEXT_RESULTS,
                )
            )
        )

        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else int(
                os.getenv(
                    "MAX_CONTEXT_CHARS",
                    self.DEFAULT_MAX_CONTEXT_CHARS,
                )
            )
        )

        # -----------------------------------------------------
        # Exact chunks sent to LLM
        #
        # This is important for source propagation.
        # -----------------------------------------------------

        self.last_context_results: List[Any] = []

        print()
        print("RAG Chatbot")
        print(f"LLM Model: {self.model}")
        print(
            f"Max context results: "
            f"{self.max_context_results}"
        )
        print(
            f"Max context chars: "
            f"{self.max_context_chars}"
        )

    # =========================================================
    # ASK
    # =========================================================

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.

        Returns:

            {
                "answer": str,
                "results": list,
                "sources": list
            }
        """

        if not question or not question.strip():
            return {
                "answer": self.FALLBACK_MESSAGE,
                "results": [],
                "sources": [],
            }

        print()
        print("=" * 70)
        print("RAG QUERY")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        # -----------------------------------------------------
        # 1. RETRIEVE
        # -----------------------------------------------------

        print()
        print("1. Retrieving documents...")

        results = self.retriever.retrieve(
            question
        )

        if not results:
            return {
                "answer": self.FALLBACK_MESSAGE,
                "results": [],
                "sources": [],
            }

        print(
            f"Retrieved: {len(results)}"
        )

        # -----------------------------------------------------
        # 2. RERANK
        # -----------------------------------------------------

        print()
        print("2. Reranking documents...")

        reranked = self._rerank(
            question,
            results,
        )

        if not reranked:
            return {
                "answer": self.FALLBACK_MESSAGE,
                "results": [],
                "sources": [],
            }

        print(
            f"Reranked: {len(reranked)}"
        )

        # -----------------------------------------------------
        # 3. BUILD CONTEXT
        # -----------------------------------------------------

        print()
        print("3. Building context...")

        context, selected_results = (
            self.build_context(
                reranked
            )
        )

        # -----------------------------------------------------
        # IMPORTANT:
        # Store EXACT results used by LLM.
        # -----------------------------------------------------

        self.last_context_results = (
            selected_results
        )

        print(
            f"Context results: "
            f"{len(selected_results)}"
        )

        print(
            f"Context characters: "
            f"{len(context):,}"
        )

        if not context.strip():
            return {
                "answer": self.FALLBACK_MESSAGE,
                "results": [],
                "sources": [],
            }

        # -----------------------------------------------------
        # 4. PROMPT
        # -----------------------------------------------------

        print()
        print("4. Building prompt...")

        prompt = (
            self.prompt_builder.build_prompt(
                question=question,
                context=context,
            )
        )

        print(
            f"Prompt characters: "
            f"{len(prompt):,}"
        )

        # -----------------------------------------------------
        # 5. GENERATE
        # -----------------------------------------------------

        print()
        print("5. Generating answer...")

        answer = self._generate(
            prompt
        )

        if not answer:
            answer = self.FALLBACK_MESSAGE

        # -----------------------------------------------------
        # 6. SOURCES
        # -----------------------------------------------------

        sources = self.build_sources(
            selected_results
        )

        return {
            "answer": answer,
            "results": selected_results,
            "sources": sources,
        }

    # =========================================================
    # STREAM
    # =========================================================

    def stream(
        self,
        question: str,
    ) -> Iterator[str]:
        """
        Run the RAG pipeline and stream the answer.

        The method performs:

            retrieve
                ↓
            rerank
                ↓
            context
                ↓
            prompt
                ↓
            Ollama streaming
        """

        if not question or not question.strip():
            yield self.FALLBACK_MESSAGE
            return

        # -----------------------------------------------------
        # Retrieve
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print("RAG QUERY")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        print()
        print("1. Retrieving documents...")

        results = self.retriever.retrieve(
            question
        )

        if not results:
            yield self.FALLBACK_MESSAGE
            return

        print(
            f"Retrieved: {len(results)}"
        )

        # -----------------------------------------------------
        # Rerank
        # -----------------------------------------------------

        print()
        print("2. Reranking documents...")

        reranked = self._rerank(
            question,
            results,
        )

        if not reranked:
            yield self.FALLBACK_MESSAGE
            return

        print(
            f"Reranked: {len(reranked)}"
        )

        # -----------------------------------------------------
        # Context
        # -----------------------------------------------------

        print()
        print("3. Building context...")

        context, selected_results = (
            self.build_context(
                reranked
            )
        )

        self.last_context_results = (
            selected_results
        )

        print(
            f"Context results: "
            f"{len(selected_results)}"
        )

        print(
            f"Context characters: "
            f"{len(context):,}"
        )

        if not context.strip():
            yield self.FALLBACK_MESSAGE
            return

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        prompt = (
            self.prompt_builder.build_prompt(
                question=question,
                context=context,
            )
        )

        print()
        print(
            "Sending prompt to Ollama..."
        )

        print(
            f"Model      : {self.model}"
        )

        print(
            f"Results    : "
            f"{len(selected_results)}"
        )

        print(
            f"Context    : "
            f"{len(context):,} characters"
        )

        print(
            f"Prompt     : "
            f"{len(prompt):,} characters"
        )

        # -----------------------------------------------------
        # Ollama streaming
        # -----------------------------------------------------

        start_time = time.perf_counter()

        first_token_time = None

        stream = self.llm_client.stream(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        for token in stream:

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

        total_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Ollama total: "
            f"{total_time:.3f} sec"
        )

    # =========================================================
    # GENERATE COMPLETE ANSWER
    # =========================================================

    def _generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a complete answer using OllamaClient.
        """

        print()
        print(
            "Sending prompt to Ollama..."
        )

        print(
            f"Model      : {self.model}"
        )

        print(
            f"Prompt     : "
            f"{len(prompt):,} characters"
        )

        start_time = time.perf_counter()

        response = self.llm_client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Ollama total: "
            f"{elapsed:.3f} sec"
        )

        return self._extract_answer(
            response
        )

    # =========================================================
    # ANSWER FROM PRE-RETRIEVED RESULTS
    # =========================================================

    def answer(
        self,
        question: str,
        results: List[Any],
    ) -> str:
        """
        Generate an answer from already reranked results.

        Useful for testing/backward compatibility.
        """

        if not question or not question.strip():
            return self.FALLBACK_MESSAGE

        if not results:
            return self.FALLBACK_MESSAGE

        context, selected_results = (
            self.build_context(
                results
            )
        )

        self.last_context_results = (
            selected_results
        )

        if not context.strip():
            return self.FALLBACK_MESSAGE

        prompt = (
            self.prompt_builder.build_prompt(
                question=question,
                context=context,
            )
        )

        return self._generate(
            prompt
        )

    # =========================================================
    # RERANK
    # =========================================================

    def _rerank(
        self,
        question: str,
        results: List[Any],
    ) -> List[Any]:

        if not results:
            return []

        if self.reranker is None:
            return results

        return self.reranker.rerank(
            question,
            results,
            self.chunks,
        )

    # =========================================================
    # BUILD CONTEXT
    # =========================================================

    def build_context(
        self,
        results: List[Any],
    ) -> Tuple[str, List[Any]]:
        """
        Build bounded LLM context.

        Returns:

            (
                context_string,
                exact_results_used
            )

        max_context_results limits the number of chunks.

        max_context_chars limits total context characters.

        There is intentionally NO final hard truncation.

        The selection loop itself controls the limit.
        """

        if not results:
            return "", []

        selected = []

        total_chars = 0

        for result in results:

            if len(selected) >= (
                self.max_context_results
            ):
                break

            text = self._get_text(
                result
            )

            if not text:
                continue

            # -------------------------------------------------
            # Calculate remaining context capacity.
            # -------------------------------------------------

            remaining = (
                self.max_context_chars
                - total_chars
            )

            if remaining <= 0:
                break

            # -------------------------------------------------
            # Limit individual result to remaining capacity.
            # -------------------------------------------------

            if len(text) > remaining:

                text = (
                    text[:remaining]
                    .rstrip()
                )

            if not text:
                continue

            selected.append(
                (
                    result,
                    text,
                )
            )

            total_chars += len(text)

            # Separator overhead.
            total_chars += 2

            if total_chars >= (
                self.max_context_chars
            ):
                break

        # -----------------------------------------------------
        # Build final context.
        # -----------------------------------------------------

        context_parts = []

        selected_results = []

        for result, text in selected:

            metadata = self._get_metadata(
                result
            )

            header_parts = []

            # -------------------------------------------------
            # Section
            # -------------------------------------------------

            section = metadata.get(
                "section_path"
            )

            if section:

                if isinstance(
                    section,
                    list,
                ):

                    section_text = (
                        " > ".join(
                            str(item)
                            for item in section
                        )
                    )

                else:

                    section_text = str(
                        section
                    )

                header_parts.append(
                    f"Section: {section_text}"
                )

            # -------------------------------------------------
            # Page
            # -------------------------------------------------

            page_start = metadata.get(
                "page_start"
            )

            page_end = metadata.get(
                "page_end"
            )

            if page_start is not None:

                if (
                    page_end is not None
                    and page_end != page_start
                ):

                    header_parts.append(
                        f"Pages: "
                        f"{page_start}-{page_end}"
                    )

                else:

                    header_parts.append(
                        f"Page: {page_start}"
                    )

            # -------------------------------------------------
            # Add metadata only when available.
            # -------------------------------------------------

            if header_parts:

                context_parts.append(
                    "\n".join(
                        header_parts
                    )
                )

            context_parts.append(
                text
            )

            # -------------------------------------------------
            # EXACT result used by LLM.
            # -------------------------------------------------

            selected_results.append(
                result
            )

        context = "\n\n".join(
            context_parts
        )

        return (
            context,
            selected_results,
        )

    # =========================================================
    # BUILD SOURCES
    # =========================================================

    def build_sources(
        self,
        results: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Build source metadata from the exact results
        used as LLM context.
        """

        if not results:
            return []

        sources = []

        seen = set()

        for result in results:

            metadata = self._get_metadata(
                result
            )

            chunk_id = (
                metadata.get(
                    "chunk_id"
                )
                or metadata.get(
                    "id"
                )
                or self._get_value(
                    result,
                    "chunk_id",
                )
            )

            page_start = metadata.get(
                "page_start"
            )

            page_end = metadata.get(
                "page_end"
            )

            section_path = metadata.get(
                "section_path",
                [],
            )

            key = (
                chunk_id,
                page_start,
                page_end,
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "chunk_id": chunk_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_path": section_path,
                }
            )

        return sources

    # =========================================================
    # TEXT HELPER
    # =========================================================

    def _get_text(
        self,
        result,
    ) -> str:

        # -----------------------------------------------------
        # LangChain Document style
        # -----------------------------------------------------

        value = self._get_value(
            result,
            "page_content",
            None,
        )

        if value:
            return str(
                value
            ).strip()

        # -----------------------------------------------------
        # Dictionary style
        # -----------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            value = (
                result.get("text")
                or result.get("content")
            )

            if value:
                return str(
                    value
                ).strip()

        return ""

    # =========================================================
    # METADATA HELPER
    # =========================================================

    def _get_metadata(
        self,
        result,
    ) -> Dict[str, Any]:

        metadata = self._get_value(
            result,
            "metadata",
            None,
        )

        if isinstance(
            metadata,
            dict,
        ):
            return metadata

        if isinstance(
            result,
            dict,
        ):

            value = result.get(
                "metadata"
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        return {}

    # =========================================================
    # GENERIC VALUE HELPER
    # =========================================================

    def _get_value(
        self,
        obj,
        attribute,
        default=None,
    ):

        if obj is None:
            return default

        if isinstance(
            obj,
            dict,
        ):
            return obj.get(
                attribute,
                default,
            )

        return getattr(
            obj,
            attribute,
            default,
        )

    # =========================================================
    # RESPONSE EXTRACTION
    # =========================================================

    def _extract_answer(
        self,
        response,
    ) -> str:

        if response is None:
            return ""

        if isinstance(
            response,
            str,
        ):
            return response.strip()

        if isinstance(
            response,
            dict,
        ):

            for key in (
                "response",
                "answer",
                "content",
                "text",
            ):

                value = response.get(
                    key
                )

                if value:
                    return str(
                        value
                    ).strip()

        value = getattr(
            response,
            "response",
            None,
        )

        if value:
            return str(
                value
            ).strip()

        value = getattr(
            response,
            "content",
            None,
        )

        if value:
            return str(
                value
            ).strip()

        return str(
            response
        ).strip()