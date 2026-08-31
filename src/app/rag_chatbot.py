import os
import time
from typing import Any, Dict, List, Tuple, Iterator
from src.app.conversation_memory import ConversationMemory

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
        Azure OpenAI
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

    DEFAULT_MODEL = "gpt-5.4-mini"
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
        openapi_client,
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
        self.openapi_client = openapi_client
        self.chunks = chunks
        self.memory = ConversationMemory(
            max_interactions=4
        )

        # -----------------------------------------------------
        # Configuration
        # -----------------------------------------------------

        self.model = (
            model
            or os.getenv(
                "AZURE_OPENAI_CHAT_DEPLOYMENT",
                self.DEFAULT_MODEL,
            )
        )

        self.temperature = (
            temperature
            if temperature is not None
            else float(
                os.getenv(
                    "AZURE_OPENAI_TEMPERATURE",
                    self.DEFAULT_TEMPERATURE,
                )
            )
        )

        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(
                os.getenv(
                    "AZURE_OPENAI_MAX_TOKENS",
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

        if not self.model:
            raise ValueError(
                "AZURE_OPENAI_CHAT_DEPLOYMENT is not configured."
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
    # SELECT CONTEXT
    # =========================================================

    def _select_context(
        self,
        results
    ):
        """
        Select the final results used as LLM context.

        Selection is limited by:

            1. Maximum number of context results
            2. Maximum context characters

        The ranking/order produced by the reranker is preserved.
        """

        selected = []

        total_chars = 0

        for result in results:

            text = result.get(
                "text",
                ""
            )

            if not text:
                continue

            text_length = len(text)

            # -------------------------------------------------
            # Maximum result count
            # -------------------------------------------------

            if len(selected) >= (
                self.max_context_results
            ):
                break

            # -------------------------------------------------
            # Maximum context characters
            # -------------------------------------------------

            if (
                total_chars + text_length
                > self.max_context_chars
            ):

                # Do not partially cut a result.
                break

            selected.append(
                result
            )

            total_chars += text_length

        return selected

    # =========================================================
    # ASK
    # =========================================================

    def ask(
        self,
        question
    ):

        print()
        print("=" * 70)
        print("RAG QUERY")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        # =====================================================
        # 1. GET CONVERSATION MEMORY
        # =====================================================

        conversation_history = (
            self.memory.format_history()
        )

        print()
        print(
            f"Memory interactions: "
            f"{self.memory.count()}"
        )

        # =====================================================
        # 2. RETRIEVE
        # =====================================================

        print()
        print("1. Retrieving documents...")

        results = self.retriever.retrieve(
            question
        )

        print(
            f"Retrieved: {len(results)}"
        )

        # =====================================================
        # 3. RERANK
        # =====================================================

        print()
        print("2. Reranking documents...")

        reranked = self._rerank(
            question,
            results
        )

        print(
            f"Reranked: {len(reranked)}"
        )

        # =====================================================
        # 4. BUILD CONTEXT
        # =====================================================

        print()
        print("3. Building context...")

        context_results = (
            self._select_context(
                reranked
            )
        )

        context = self.build_context(
            context_results
        )

        print(
            f"Context results: "
            f"{len(context_results)}"
        )

        print(
            f"Context characters: "
            f"{len(context)}"
        )

        # =====================================================
        # 5. BUILD PROMPT
        # =====================================================

        print()
        print("4. Building prompt...")

        prompt = (
            self.prompt_builder.build_prompt(
                question=question,
                context=context,
                conversation_history=(
                    conversation_history
                )
            )
        )

        print(
            f"Prompt characters: "
            f"{len(prompt)}"
        )

        # =====================================================
        # 6. GENERATE ANSWER
        # =====================================================

        print()
        print("5. Generating answer...")

        answer = self.openapi_client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # =====================================================
        # 7. SAVE MEMORY
        # =====================================================

        self.memory.add(
            question=question,
            answer=answer
        )

        # =====================================================
        # 8. SOURCES
        # =====================================================

        sources = self.build_sources(
            context_results
        )

        return {
            "answer": answer,
            "results": context_results,
            "sources": sources
        }

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
            Azure OpenAI streaming
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
            "Sending prompt to Azure OpenAI..."
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
        # LLM streaming / compatibility
        # -----------------------------------------------------

        start_time = time.perf_counter()

        first_token_time = None

        if hasattr(self.openapi_client, "stream"):
            response_stream = self.openapi_client.stream(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            for token in response_stream:

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
        else:
            response = self.openapi_client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = self._extract_answer(response)

            if answer:
                first_token_time = (
                    time.perf_counter()
                    - start_time
                )
                print(
                    f"First token: "
                    f"{first_token_time:.3f} sec"
                )
                yield answer

        total_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"LLM total: "
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
        Generate a complete answer using the configured LLM client.
        """

        print()
        print(
            "Sending prompt to Azure OpenAI..."
        )

        print(
            f"Model      : {self.model}"
        )

        print(
            f"Prompt     : "
            f"{len(prompt):,} characters"
        )

        start_time = time.perf_counter()

        response = self.openapi_client.generate(
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
            f"LLM total: "
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
                []
            )

            document_name = metadata.get(
                "document_name"
            )

            document_id = metadata.get(
                "document_id"
            )

            key = (
                document_id,
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

                    "document_id":
                        document_id,

                    "document_name":
                        document_name,

                    "page_start":
                        page_start,

                    "page_end":
                        page_end,

                    "section_path":
                        section_path,
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
        """
        Resolve metadata from a retrieval result.

        Supports both:

            {
                "metadata": {
                    "page_start": 1,
                    "page_end": 2
                }
            }

        and:

            {
                "page_start": 1,
                "page_end": 2
            }

        Top-level fields are merged into nested metadata
        without overwriting valid nested values.
        """

        metadata = {}

        # ---------------------------------------------------------
        # Nested metadata
        # ---------------------------------------------------------

        nested = self._get_value(
            result,
            "metadata",
            None,
        )

        if isinstance(
            nested,
            dict,
        ):

            metadata.update(
                nested
            )

        # ---------------------------------------------------------
        # Top-level metadata
        # ---------------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            metadata_fields = (
                "chunk_id",
                "document_id",
                "document_name",
                "content_type",
                "page_start",
                "page_end",
                "section_path",
                "caption",
                "document_part",
            )

            for field in metadata_fields:

                value = result.get(
                    field
                )

                if value is not None:

                    # Top-level value is used only
                    # when nested metadata doesn't
                    # already contain a value.
                    if (
                        field not in metadata
                        or metadata[field] is None
                    ):

                        metadata[field] = value

        # ---------------------------------------------------------
        # Object attributes
        # ---------------------------------------------------------

        else:

            metadata_fields = (
                "chunk_id",
                "document_id",
                "document_name",
                "content_type",
                "page_start",
                "page_end",
                "section_path",
                "caption",
                "document_part",
            )

            for field in metadata_fields:

                value = getattr(
                    result,
                    field,
                    None,
                )

                if value is not None:

                    if (
                        field not in metadata
                        or metadata[field] is None
                    ):

                        metadata[field] = value

        return metadata

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