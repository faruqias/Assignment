from pathlib import Path
import json
import re

from document_parser import DocumentParser, Element


class StructureChunker:
    """
    Structure-aware chunker for the new RAG ingestion pipeline.

    Pipeline:

        Docling document
              ↓
        DocumentParser
              ↓
        normalized Elements
              ↓
        StructureChunker
              ↓
        text / table / figure chunks

    This class does NOT handle:

        - embeddings
        - FAISS
        - BM25
        - RRF
        - reranking
        - LLM generation
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.parser = DocumentParser()

    # ============================================================
    # PUBLIC API
    # ============================================================

    def chunk(self, document):
        """
        Create structure-aware chunks.

        Parameters
        ----------
        document:
            Can be:

            - document.json path
            - Path object
            - loaded JSON dictionary
            - list of normalized Elements
            - live Docling document

        Returns
        -------
        list[dict]
        """

        print()
        print("=" * 70)
        print("STRUCTURE CHUNKER")
        print("=" * 70)

        # --------------------------------------------------------
        # Already parsed elements
        # --------------------------------------------------------

        if isinstance(document, list):

            elements = document

        # --------------------------------------------------------
        # Live Docling document
        # --------------------------------------------------------

        elif hasattr(
            document,
            "iterate_items"
        ):

            elements = self._parse_docling_document(
                document
            )

        # --------------------------------------------------------
        # document.json
        # --------------------------------------------------------

        else:

            elements = self.parser.parse(
                document
            )

        print(
            f"Elements received : {len(elements)}"
        )

        chunks = self._build_chunks(
            elements
        )

        self._assign_chunk_ids(
            chunks
        )

        print()
        print(
            f"Chunks created    : {len(chunks)}"
        )

        self._print_distribution(
            chunks
        )

        return chunks

    # ============================================================
    # LIVE DOCLING DOCUMENT
    # ============================================================

    def _parse_docling_document(
        self,
        document
    ):

        """
        Converts a live Docling Document into normalized
        Element objects.

        Normally DocumentParser is used instead.
        This method allows the chunker to also accept
        a live Docling document.
        """

        elements = []

        section_path = []

        document_part = "main"

        index = 0

        for item, _level in document.iterate_items():

            item_type = (
                item.__class__.__name__
            )

            text = ""

            if hasattr(
                item,
                "text"
            ):

                text = (
                    item.text or ""
                )

                text = self._clean_text(
                    text
                )

            pages = self._get_pages(
                item
            )

            page_start = (
                min(pages)
                if pages
                else None
            )

            page_end = (
                max(pages)
                if pages
                else None
            )

            # ----------------------------------------------------
            # Heading
            # ----------------------------------------------------

            if item_type in {
                "SectionHeaderItem",
                "TitleItem"
            }:

                if text:

                    section_path = (
                        self._update_section_path(
                            section_path,
                            text
                        )
                    )

                    elements.append(
                        Element(
                            element_type="heading",
                            index=index,
                            text=text,
                            item=None,
                            section_path=section_path.copy(),
                            page_start=page_start,
                            page_end=page_end,
                            document_part=document_part
                        )
                    )

                index += 1

                continue

            # ----------------------------------------------------
            # Picture
            # ----------------------------------------------------

            if item_type == "PictureItem":

                elements.append(
                    Element(
                        element_type="picture",
                        index=index,
                        text=text,
                        item=item,
                        section_path=section_path.copy(),
                        page_start=page_start,
                        page_end=page_end,
                        document_part=document_part
                    )
                )

                index += 1

                continue

            # ----------------------------------------------------
            # Table
            # ----------------------------------------------------

            if item_type == "TableItem":

                elements.append(
                    Element(
                        element_type="table",
                        index=index,
                        text=text,
                        item=item,
                        section_path=section_path.copy(),
                        page_start=page_start,
                        page_end=page_end,
                        document_part=document_part
                    )
                )

                index += 1

                continue

            # ----------------------------------------------------
            # Text
            # ----------------------------------------------------

            if text:

                elements.append(
                    Element(
                        element_type="text",
                        index=index,
                        text=text,
                        item=None,
                        section_path=section_path.copy(),
                        page_start=page_start,
                        page_end=page_end,
                        document_part=document_part
                    )
                )

            index += 1

        return elements

    # ============================================================
    # BUILD CHUNKS
    # ============================================================

    def _build_chunks(
        self,
        elements
    ):

        chunks = []

        text_buffer = []

        text_section = []

        text_page_start = None

        text_page_end = None

        text_document_part = "main"

        # --------------------------------------------------------
        # Flush text buffer
        # --------------------------------------------------------

        def flush_text():

            nonlocal text_buffer
            nonlocal text_section
            nonlocal text_page_start
            nonlocal text_page_end
            nonlocal text_document_part

            if not text_buffer:

                return

            combined_text = "\n\n".join(
                text_buffer
            )

            parts = self._split_text(
                combined_text
            )

            for part in parts:

                if not part.strip():

                    continue

                chunks.append(
                    self._create_text_chunk(
                        text=part,
                        section_path=text_section,
                        page_start=text_page_start,
                        page_end=text_page_end,
                        document_part=text_document_part
                    )
                )

            text_buffer = []

            text_section = []

            text_page_start = None

            text_page_end = None

            text_document_part = "main"

        # --------------------------------------------------------
        # Process elements
        # --------------------------------------------------------

        for element in elements:

            # ====================================================
            # HEADING
            # ====================================================

            if element.element_type == "heading":

                flush_text()

                continue

            # ====================================================
            # TEXT
            # ====================================================

            if element.element_type == "text":

                text = self._clean_text(
                    element.text
                )

                if not text:

                    continue

                if not text_buffer:

                    text_section = (
                        element.section_path.copy()
                    )

                    text_page_start = (
                        element.page_start
                    )

                    text_page_end = (
                        element.page_end
                    )

                    text_document_part = (
                        element.document_part
                    )

                else:

                    if (
                        element.page_end
                        is not None
                    ):

                        text_page_end = (
                            element.page_end
                        )

                text_buffer.append(
                    text
                )

                continue

            # ====================================================
            # TABLE
            # ====================================================

            if element.element_type == "table":

                flush_text()

                chunk = (
                    self._create_table_chunk(
                        element
                    )
                )

                if chunk:

                    chunks.append(
                        chunk
                    )

                continue

            # ====================================================
            # PICTURE
            # ====================================================

            if element.element_type == "picture":

                flush_text()

                chunk = (
                    self._create_figure_chunk(
                        element
                    )
                )

                if chunk:

                    chunks.append(
                        chunk
                    )

                continue

        # --------------------------------------------------------
        # Remaining text
        # --------------------------------------------------------

        flush_text()

        return chunks

    # ============================================================
    # TEXT CHUNK
    # ============================================================

    def _create_text_chunk(
        self,
        text,
        section_path,
        page_start,
        page_end,
        document_part
    ):

        return {

            "chunk_id": None,

            "content_type": "text",

            "text": text,

            "caption": None,

            "page_start": page_start,

            "page_end": page_end,

            "section_path": section_path.copy(),

            "section": (
                section_path[-1]
                if section_path
                else None
            ),

            "document_part": document_part,

            "token_count": len(
                text.split()
            ),

            "parent_id": None,

            "is_atomic": False,

            "image_path": None,

            "referenced_from": []
        }

    # ============================================================
    # TABLE CHUNK
    # ============================================================

    def _create_table_chunk(
        self,
        element
    ):

        item = element.item

        if item is None:

            return None

        caption = self._get_caption(
            item
        )

        table_text = self._get_table_text(
            item
        )

        return {

            "chunk_id": None,

            "content_type": "table",

            "text": table_text,

            "caption": caption,

            "page_start": (
                element.page_start
            ),

            "page_end": (
                element.page_end
            ),

            "section_path": (
                element.section_path.copy()
            ),

            "section": (
                element.section_path[-1]
                if element.section_path
                else None
            ),

            "document_part": (
                element.document_part
            ),

            "token_count": (
                len(table_text.split())
                if table_text
                else 0
            ),

            "parent_id": None,

            "is_atomic": True,

            "image_path": None,

            "referenced_from": []
        }

    # ============================================================
    # FIGURE CHUNK
    # ============================================================

    def _create_figure_chunk(
        self,
        element
    ):

        item = element.item

        if item is None:

            return None

        caption = self._get_caption(
            item
        )

        return {

            "chunk_id": None,

            "content_type": "figure",

            "text": "",

            "caption": caption,

            "page_start": (
                element.page_start
            ),

            "page_end": (
                element.page_end
            ),

            "section_path": (
                element.section_path.copy()
            ),

            "section": (
                element.section_path[-1]
                if element.section_path
                else None
            ),

            "document_part": (
                element.document_part
            ),

            "token_count": (
                len(caption.split())
                if caption
                else 0
            ),

            "parent_id": None,

            "is_atomic": True,

            "image_path": (
                self._get_image_path(
                    item
                )
            ),

            "referenced_from": []
        }

    # ============================================================
    # CAPTION
    # ============================================================

    def _get_caption(
        self,
        item
    ):

        try:

            captions = getattr(
                item,
                "captions",
                None
            )

            if not captions:

                return None

            values = []

            for caption in captions:

                if hasattr(
                    caption,
                    "text"
                ):

                    values.append(
                        caption.text
                    )

                else:

                    values.append(
                        str(caption)
                    )

            result = " ".join(
                values
            )

            result = self._clean_text(
                result
            )

            return result or None

        except Exception:

            return None

    # ============================================================
    # TABLE TEXT
    # ============================================================

    def _get_table_text(
        self,
        item
    ):

        # --------------------------------------------------------
        # DataFrame
        # --------------------------------------------------------

        try:

            if hasattr(
                item,
                "export_to_dataframe"
            ):

                dataframe = (
                    item.export_to_dataframe()
                )

                return self._clean_text(
                    dataframe.to_string(
                        index=False
                    )
                )

        except Exception:

            pass

        # --------------------------------------------------------
        # Direct text
        # --------------------------------------------------------

        try:

            text = getattr(
                item,
                "text",
                ""
            )

            if text:

                return self._clean_text(
                    text
                )

        except Exception:

            pass

        return ""

    # ============================================================
    # IMAGE PATH
    # ============================================================

    def _get_image_path(
        self,
        item
    ):

        try:

            image = getattr(
                item,
                "image",
                None
            )

            if image is None:

                return None

            uri = getattr(
                image,
                "uri",
                None
            )

            if uri:

                return str(uri)

        except Exception:

            pass

        return None

    # ============================================================
    # SPLIT TEXT
    # ============================================================

    def _split_text(
        self,
        text
    ):

        text = self._clean_text(
            text
        )

        if not text:

            return []

        words = text.split()

        # --------------------------------------------------------
        # Small enough
        # --------------------------------------------------------

        if len(words) <= self.chunk_size:

            return [text]

        chunks = []

        start = 0

        while start < len(words):

            end = (
                start
                + self.chunk_size
            )

            part = " ".join(
                words[start:end]
            )

            if part.strip():

                chunks.append(
                    part
                )

            if end >= len(words):

                break

            start = max(
                0,
                end - self.chunk_overlap
            )

        return chunks

    # ============================================================
    # CHUNK IDS
    # ============================================================

    def _assign_chunk_ids(
        self,
        chunks
    ):

        counters = {
            "text": 0,
            "table": 0,
            "figure": 0
        }

        for chunk in chunks:

            content_type = (
                chunk["content_type"]
            )

            number = counters.get(
                content_type,
                0
            )

            chunk["chunk_id"] = (
                f"chunk_{content_type}_{number}"
            )

            counters[
                content_type
            ] = number + 1

    # ============================================================
    # PAGE EXTRACTION
    # ============================================================

    def _get_pages(
        self,
        item
    ):

        pages = []

        try:

            prov = getattr(
                item,
                "prov",
                []
            )

            for value in prov:

                page = getattr(
                    value,
                    "page_no",
                    None
                )

                if page is not None:

                    pages.append(
                        int(page)
                    )

        except Exception:

            pass

        return sorted(
            set(pages)
        )

    # ============================================================
    # SECTION PATH
    # ============================================================

    def _update_section_path(
        self,
        current_path,
        title
    ):

        title = self._clean_text(
            title
        )

        match = re.match(
            r"^(\d+(?:\.\d+)*)\s+",
            title
        )

        if not match:

            return current_path

        depth = len(
            match.group(1).split(".")
        )

        return (
            current_path[
                :depth - 1
            ]
            + [title]
        )

    # ============================================================
    # CLEAN TEXT
    # ============================================================

    @staticmethod
    def _clean_text(
        text
    ):

        if not text:

            return ""

        text = str(
            text
        )

        text = text.replace(
            "\u00a0",
            " "
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # ============================================================
    # DISTRIBUTION
    # ============================================================

    def _print_distribution(
        self,
        chunks
    ):

        distribution = {}

        for chunk in chunks:

            content_type = (
                chunk.get(
                    "content_type",
                    "unknown"
                )
            )

            distribution[
                content_type
            ] = (
                distribution.get(
                    content_type,
                    0
                )
                + 1
            )

        print()
        print(
            "Content distribution:"
        )

        for content_type, count in (
            distribution.items()
        ):

            print(
                f"  {content_type:<10}: "
                f"{count}"
            )

    # ============================================================
    # SAVE
    # ============================================================

    def save_chunks(
        self,
        chunks,
        output_path
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Chunks saved: {output_path}"
        )