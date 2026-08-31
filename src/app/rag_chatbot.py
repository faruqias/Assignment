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
        Conversation Memory
            ↓
        Hybrid Retriever
            ↓
        BGE Reranker (optional)
            ↓
        Context Selection
            ↓
        Prompt Builder
            ↓
        Azure OpenAI
            ↓
        Answer
            ↓
        Conversation Memory
            ↓
        Sources

    Responsibilities:
        - Retrieve relevant documents
        - Rerank documents
        - Select final context
        - Build prompt
        - Generate answer
        - Stream answer
        - Maintain conversation memory
        - Track exact source chunks used

    Does NOT handle:
        - PDF processing
        - Parsing
        - Chunking
        - Embedding generation
        - FAISS creation
        - BM25 creation
        - RRF creation
    """

    DEFAULT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini")
    DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    DEFAULT_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "500"))
    DEFAULT_MAX_CONTEXT_RESULTS = int(os.getenv("MAX_CONTEXT_RESULTS", "3"))
    DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))

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

        # =====================================================
        # DEPENDENCIES
        # =====================================================

        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.openapi_client = openapi_client
        self.chunks = chunks

        # =====================================================
        # CONVERSATION MEMORY
        # =====================================================

        self.memory = ConversationMemory(
            max_interactions=4
        )

        # =====================================================
        # CONFIGURATION
        # =====================================================

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
                "AZURE_OPENAI_CHAT_DEPLOYMENT "
                "is not configured."
            )

        # =====================================================
        # EXACT RESULTS USED BY LLM
        # =====================================================

        self.last_context_results: List[Any] = []

        # =====================================================
        # STARTUP INFORMATION
        # =====================================================

        print()
        print("RAG Chatbot")

        print(
            f"LLM Model: {self.model}"
        )

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

        # =====================================================
        # 1. CONVERSATION MEMORY
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

        if not results:

            print(
                "No retrieval results found."
            )

            return {
                "answer": self.FALLBACK_MESSAGE,
                "results": [],
                "sources": [],
            }

        # =====================================================
        # 3. RERANK
        # =====================================================

        print()
        print("2. Reranking documents...")

        reranked = self._rerank(
            question,
            results,
        )

        print(
            f"Reranked: {len(reranked)}"
        )

        if not reranked:

            return {
                "answer": self.FALLBACK_MESSAGE,
                "results": [],
                "sources": [],
            }

        # =====================================================
        # 4. SELECT CONTEXT
        # =====================================================

        print()
        print("3. Building context...")

        context_results = (
            self._select_context(
                reranked
            )
        )

        # Store exact chunks used by LLM.
        self.last_context_results = (
            context_results
        )

        # =====================================================
        # 5. BUILD CONTEXT
        # =====================================================

        context = self.build_context(
            context_results
        )

        print(
            f"Context results: "
            f"{len(context_results)}"
        )

        print(
            f"Context characters: "
            f"{len(context):,}"
        )

        if not context.strip():

            self.last_context_results = []

            return {
                "answer": self.FALLBACK_MESSAGE,
                "results": [],
                "sources": [],
            }

        # =====================================================
        # 6. BUILD PROMPT
        # =====================================================

        print()
        print("4. Building prompt...")

        prompt = (
            self.prompt_builder.build_prompt(
                question=question,
                context=context,
                conversation_history=(
                    conversation_history
                ),
            )
        )

        print(
            f"Prompt characters: "
            f"{len(prompt):,}"
        )

        # =====================================================
        # 7. GENERATE ANSWER
        # =====================================================

        print()
        print("5. Generating answer...")

        answer = self._generate(
            prompt
        )

        if not answer:

            answer = self.FALLBACK_MESSAGE

        # =====================================================
        # 8. SAVE CONVERSATION MEMORY
        # =====================================================

        self.memory.add(
            question=question,
            answer=answer,
        )

        # =====================================================
        # 9. SOURCES
        # =====================================================

        sources = self.build_sources(
            context_results
        )

        return {
            "answer": answer,
            "results": context_results,
            "sources": sources,
        }

    # =========================================================
    # STREAM
    # =========================================================

    def stream(
        self,
        question: str,
    ) -> Iterator[str]:

        if not question or not question.strip():

            yield self.FALLBACK_MESSAGE
            return

        print()
        print("=" * 70)
        print("RAG QUERY")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        # =====================================================
        # 1. MEMORY
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

        if not results:

            yield self.FALLBACK_MESSAGE
            return

        # =====================================================
        # 3. RERANK
        # =====================================================

        print()
        print("2. Reranking documents...")

        reranked = self._rerank(
            question,
            results,
        )

        print(
            f"Reranked: {len(reranked)}"
        )

        if not reranked:

            yield self.FALLBACK_MESSAGE
            return

        # =====================================================
        # 4. SELECT CONTEXT
        # =====================================================

        print()
        print("3. Building context...")

        selected_results = (
            self._select_context(
                reranked
            )
        )

        self.last_context_results = (
            selected_results
        )

        # =====================================================
        # 5. BUILD CONTEXT
        # =====================================================

        context = self.build_context(
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

            self.last_context_results = []

            yield self.FALLBACK_MESSAGE
            return

        # =====================================================
        # 6. BUILD PROMPT
        # =====================================================

        print()
        print("4. Building prompt...")

        prompt = (
            self.prompt_builder.build_prompt(
                question=question,
                context=context,
                conversation_history=(
                    conversation_history
                ),
            )
        )

        print(
            f"Prompt characters: "
            f"{len(prompt):,}"
        )

        # =====================================================
        # 7. STREAM FROM LLM
        # =====================================================

        print()
        print(
            "5. Generating answer..."
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

        start_time = time.perf_counter()

        first_token_time = None

        complete_answer = ""

        # =====================================================
        # STREAMING CLIENT
        # =====================================================

        if hasattr(
            self.openapi_client,
            "stream",
        ):

            response_stream = (
                self.openapi_client.stream(
                    prompt=prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )

            for token in response_stream:

                if not token:
                    continue

                token = str(token)

                complete_answer += token

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

        # =====================================================
        # FALLBACK TO GENERATE()
        # =====================================================

        else:

            response = (
                self.openapi_client.generate(
                    prompt=prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )

            complete_answer = (
                self._extract_answer(
                    response
                )
            )

            if complete_answer:

                first_token_time = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"First token: "
                    f"{first_token_time:.3f} sec"
                )

                yield complete_answer

        total_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"LLM total: "
            f"{total_time:.3f} sec"
        )

        # =====================================================
        # SAVE MEMORY AFTER STREAM COMPLETES
        # =====================================================

        if complete_answer.strip():

            self.memory.add(
                question=question,
                answer=complete_answer,
            )

    # =========================================================
    # SELECT CONTEXT
    # =========================================================

    def _select_context(
        self,
        results: List[Any],
    ) -> List[Any]:
        """
        Select complete chunks for the LLM.

        Limits:

            max_context_results
            max_context_chars

        A chunk is either included completely
        or excluded.

        No chunk is partially truncated.
        """

        if not results:
            return []

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
            # Account for separator between chunks.
            # -------------------------------------------------

            separator_chars = (
                2 if selected else 0
            )

            required_chars = (
                separator_chars
                + len(text)
            )

            if (
                total_chars
                + required_chars
                > self.max_context_chars
            ):

                # Do not partially truncate
                # a chunk.
                continue

            selected.append(
                result
            )

            total_chars += required_chars

        return selected

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

        # -----------------------------------------------------
        # No reranker configured
        # -----------------------------------------------------

        if self.reranker is None:

            print(
                "BGE Reranker: SKIPPED"
            )

            return results

        # -----------------------------------------------------
        # Reranker handles enabled/disabled
        # internally.
        # -----------------------------------------------------

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
    ) -> str:
        """
        Format already-selected results.

        IMPORTANT:

        This method does NOT perform selection.

        _select_context() is responsible for:

            max_context_results
            max_context_chars

        This method only formats the selected chunks.

        There is intentionally NO final hard truncation.
        """

        if not results:
            return ""

        context_parts = []

        for result in results:

            text = self._get_text(
                result
            )

            if not text:
                continue

            metadata = self._get_metadata(
                result
            )

            header_parts = []

            # -------------------------------------------------
            # Document
            # -------------------------------------------------

            document_name = metadata.get(
                "document_name"
            )

            if document_name:

                header_parts.append(
                    f"Document: "
                    f"{document_name}"
                )

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
                    f"Section: "
                    f"{section_text}"
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
                        f"{page_start}-"
                        f"{page_end}"
                    )

                else:

                    header_parts.append(
                        f"Page: "
                        f"{page_start}"
                    )

            # -------------------------------------------------
            # Chunk
            # -------------------------------------------------

            chunk_id = metadata.get(
                "chunk_id"
            )

            if chunk_id:

                header_parts.append(
                    f"Chunk: "
                    f"{chunk_id}"
                )

            # -------------------------------------------------
            # Build block
            # -------------------------------------------------

            block_parts = []

            if header_parts:

                block_parts.append(
                    "\n".join(
                        header_parts
                    )
                )

            block_parts.append(
                text
            )

            context_parts.append(
                "\n".join(
                    block_parts
                )
            )

        return "\n\n".join(
            context_parts
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
        Generate an answer from already-retrieved
        or already-reranked results.

        Useful for tests/backward compatibility.
        """

        if not question or not question.strip():

            return self.FALLBACK_MESSAGE

        if not results:

            return self.FALLBACK_MESSAGE

        # -----------------------------------------------------
        # Select context
        # -----------------------------------------------------

        selected_results = (
            self._select_context(
                results
            )
        )

        self.last_context_results = (
            selected_results
        )

        # -----------------------------------------------------
        # Build context
        # -----------------------------------------------------

        context = self.build_context(
            selected_results
        )

        if not context.strip():

            return self.FALLBACK_MESSAGE

        # -----------------------------------------------------
        # Conversation history
        # -----------------------------------------------------

        conversation_history = (
            self.memory.format_history()
        )

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        prompt = (
            self.prompt_builder.build_prompt(
                question=question,
                context=context,
                conversation_history=(
                    conversation_history
                ),
            )
        )

        # -----------------------------------------------------
        # Generate
        # -----------------------------------------------------

        answer = self._generate(
            prompt
        )

        if not answer:

            return self.FALLBACK_MESSAGE

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        self.memory.add(
            question=question,
            answer=answer,
        )

        return answer

    # =========================================================
    # GENERATE COMPLETE ANSWER
    # =========================================================

    def _generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a complete answer using the
        configured LLM client.
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

        response = (
            self.openapi_client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Azure OpenAI total: "
            f"{elapsed:.3f} sec"
        )

        return self._extract_answer(
            response
        )

    # =========================================================
    # BUILD SOURCES
    # =========================================================

    def build_sources(
        self,
        results: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Build source metadata from the exact
        chunks used as LLM context.
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

            document_id = metadata.get(
                "document_id"
            )

            document_name = metadata.get(
                "document_name"
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

            key = (
                document_id,
                document_name,
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
        # LangChain Document
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
        # Dictionary
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

        Supports:

            {
                "metadata": {...}
            }

        and:

            {
                "page_start": 4,
                "page_end": 4
            }

        Nested metadata takes precedence over
        top-level metadata.
        """

        metadata = {}

        # =====================================================
        # NESTED METADATA
        # =====================================================

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

        # =====================================================
        # COMMON METADATA FIELDS
        # =====================================================

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

        # =====================================================
        # DICTIONARY
        # =====================================================

        if isinstance(
            result,
            dict,
        ):

            for field in metadata_fields:

                value = result.get(
                    field
                )

                if value is not None:

                    if (
                        field not in metadata
                        or metadata[field] is None
                    ):

                        metadata[field] = value

        # =====================================================
        # OBJECT
        # =====================================================

        else:

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

        # -----------------------------------------------------
        # Plain string
        # -----------------------------------------------------

        if isinstance(
            response,
            str,
        ):

            return response.strip()

        # -----------------------------------------------------
        # Dictionary
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Object
        # -----------------------------------------------------

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