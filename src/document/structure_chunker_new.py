import os
from dotenv import load_dotenv
from collections import Counter
from typing import Any, Dict, List

load_dotenv()
    
class StructureChunker:
    """
    Structure-aware document chunker.

    Responsibilities:
        - Group parsed document elements into meaningful chunks
        - Preserve section hierarchy
        - Preserve captions
        - Exclude page headers/footers
        - Exclude raw PictureItem content
        - Remove obvious diagram/OCR garbage
        - Enforce maximum chunk size

    Does NOT handle:
        - Embeddings
        - FAISS
        - BM25
        - RRF
        - Retrieval
        - Reranking
        - LLM generation
    """

    MAX_CHUNK_CHARS = int(
    os.getenv(
        "DEFAULT_CHUNK_SIZE",
        "5000"
    )
)

    # Elements that should not become normal text content.
    IGNORED_TYPES = {
        "page_header",
        "page_footer",
        "picture",
    }

    # Elements that represent structural boundaries.
    SECTION_TYPES = {
        "section_header",
    }

    # Useful non-body content that should be retained.
    CONTENT_TYPES = {
        "text",
        "list_item",
        "caption",
        "footnote",
        "table",
        "table_item",
    }

    def __init__(
        self,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
    ):
        self.max_chunk_chars = max_chunk_chars

    # =========================================================
    # PUBLIC API
    # =========================================================

    def chunk(
        self,
        elements: List[Any],
    ) -> List[Dict[str, Any]]:

        print()
        print("=" * 60)
        print("STRUCTURE CHUNKER")
        print("=" * 60)

        print(
            "Elements received :",
            len(elements),
        )

        if not elements:
            return []

        chunks = self._build_chunks(elements)

        print()
        print(
            "Chunks created    :",
            len(chunks),
        )

        distribution = Counter(
            chunk.get(
                "content_type",
                "unknown",
            )
            for chunk in chunks
        )

        print()
        print("Content distribution:")

        for content_type, count in distribution.items():

            print(
                f"  {content_type:<10}: {count}"
            )

        return chunks

    # =========================================================
    # BUILD CHUNKS
    # =========================================================

    def _build_chunks(
        self,
        elements: List[Any],
    ) -> List[Dict[str, Any]]:

        chunks = []

        current_parts = []
        current_elements = []

        current_section = []
        current_caption = None

        chunk_counter = 0

        # -----------------------------------------------------
        # Helpers
        # -----------------------------------------------------

        def flush():

            nonlocal \
                current_parts, \
                current_elements, \
                current_caption, \
                chunk_counter

            if not current_parts:
                return

            text = self._clean_text(
                "\n\n".join(current_parts)
            )

            if not text:
                current_parts = []
                current_elements = []
                current_caption = None
                return

            first_element = (
                current_elements[0]
                if current_elements
                else None
            )

            last_element = (
                current_elements[-1]
                if current_elements
                else None
            )

            chunk = {
                "chunk_id": (
                    f"chunk_text_{chunk_counter}"
                ),

                "content_type": "text",

                "text": text,

                "section_path": list(
                    current_section
                ),

                "caption": current_caption,

                "page_start": self._get_value(
                    first_element,
                    "page_start",
                ),

                "page_end": self._get_value(
                    last_element,
                    "page_end",
                ),

                "element_count": len(
                    current_elements
                ),
            }

            chunks.append(chunk)

            chunk_counter += 1

            current_parts = []
            current_elements = []
            current_caption = None

        # -----------------------------------------------------
        # Process elements
        # -----------------------------------------------------

        for element in elements:

            element_type = self._get_element_type(
                element
            )

            # -------------------------------------------------
            # Section header
            # -------------------------------------------------

            if element_type in self.SECTION_TYPES:

                # Finish previous section.
                flush()

                text = self._get_text(
                    element
                )

                if text:

                    current_section = (
                        self._get_section_path(
                            element,
                            fallback=text,
                        )
                    )

                continue

            # -------------------------------------------------
            # Ignore page headers / footers
            # -------------------------------------------------

            if element_type in {
                "page_header",
                "page_footer",
            }:

                continue

            # -------------------------------------------------
            # Ignore pictures
            # -------------------------------------------------

            if element_type in {
                "picture",
                "PictureItem",
                "picture_item",
            }:

                continue

            # -------------------------------------------------
            # Get text
            # -------------------------------------------------

            text = self._get_text(
                element
            )

            if not text:
                continue

            # -------------------------------------------------
            # Caption
            # -------------------------------------------------

            if element_type == "caption":

                # Keep caption as metadata and context.
                current_caption = text

                current_parts.append(
                    f"Caption: {text}"
                )

                current_elements.append(
                    element
                )

                continue

            # -------------------------------------------------
            # Filter diagram/OCR garbage
            # -------------------------------------------------

            if self._looks_like_diagram_garbage(
                text
            ):

                continue

            # -------------------------------------------------
            # Normal content
            # -------------------------------------------------

            current_parts.append(
                text
            )

            current_elements.append(
                element
            )

            # -------------------------------------------------
            # Enforce chunk size
            # -------------------------------------------------

            current_text = self._clean_text(
                "\n\n".join(current_parts)
            )

            if (
                len(current_text)
                >= self.max_chunk_chars
            ):

                flush()

        # -----------------------------------------------------
        # Final chunk
        # -----------------------------------------------------

        flush()

        return chunks

    # =========================================================
    # TEXT CLEANING
    # =========================================================

    def _clean_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # Collapse excessive whitespace.
            line = " ".join(
                line.split()
            )

            lines.append(line)

        return "\n\n".join(lines)

    # =========================================================
    # DIAGRAM / OCR FILTER
    # =========================================================

    def _looks_like_diagram_garbage(
        self,
        text: str,
    ) -> bool:

        normalized = (
            text
            .strip()
            .lower()
        )

        if not normalized:
            return True

        # Typical extracted diagram tokens.
        diagram_tokens = {
            "matmul",
            "softmax",
            "linear",
            "concat",
            "scale",
            "mask",
            "attention",
        }

        tokens = set(
            normalized.replace(
                "(",
                " ",
            )
            .replace(
                ")",
                " ",
            )
            .replace(
                ",",
                " ",
            )
            .split()
        )

        # Short token-only diagram strings.
        if (
            len(tokens) <= 15
            and len(
                tokens.intersection(
                    diagram_tokens
                )
            ) >= 3
        ):
            return True

        # Very high symbol density.
        alpha_count = sum(
            char.isalpha()
            for char in text
        )

        if (
            len(text) > 20
            and alpha_count / len(text)
            < 0.35
        ):
            return True

        return False

    # =========================================================
    # ELEMENT HELPERS
    # =========================================================

    def _get_element_type(
        self,
        element,
    ) -> str:

        value = self._get_value(
            element,
            "element_type",
            "",
        )

        return str(value)

    def _get_text(
        self,
        element,
    ) -> str:

        value = self._get_value(
            element,
            "text",
            "",
        )

        if value is None:
            return ""

        return str(value).strip()

    def _get_section_path(
        self,
        element,
        fallback=None,
    ) -> List[str]:

        section_path = self._get_value(
            element,
            "section_path",
            None,
        )

        if section_path:

            if isinstance(
                section_path,
                list,
            ):

                return [
                    str(x)
                    for x in section_path
                    if x
                ]

        if fallback:
            return [fallback]

        return []

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