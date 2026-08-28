from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import json
import re


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Element:

    element_type: str

    index: int

    text: str = ""

    item: Optional[dict] = None

    section_path: list[str] = field(
        default_factory=list
    )

    page_start: Optional[int] = None

    page_end: Optional[int] = None

    document_part: str = "main"


# ============================================================
# DOCUMENT PARSER
# ============================================================

class DocumentParser:

    """
    Converts Docling document.json into normalized elements.

    Responsibilities:

        document.json
              ↓
        Docling tree traversal
              ↓
        text
        headings
        tables
        pictures
              ↓
        normalized Element objects

    This class does NOT perform:

        - chunking
        - embeddings
        - FAISS indexing
        - BM25
        - RRF
        - reranking
        - LLM generation
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.document_order = 0

    # ========================================================
    # PUBLIC API
    # ========================================================

    def parse(self, document_json):

        """
        Parse a Docling document JSON.

        Parameters
        ----------
        document_json:
            Can be either:

            - Path
            - str path
            - loaded dict

        Returns
        -------
        list[Element]
        """

        root = self._load_document(
            document_json
        )

        self.document_order = 0

        elements = self._parse_elements(
            root
        )

        print(
            f"Parsed elements: {len(elements)}"
        )

        return elements

    # ========================================================
    # LOAD DOCUMENT
    # ========================================================

    def _load_document(
        self,
        document_json
    ):

        if isinstance(
            document_json,
            dict
        ):

            return document_json

        path = Path(
            document_json
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Document JSON not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    @staticmethod
    def clean_text(
        text: str
    ) -> str:

        if not text:

            return ""

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

    # ========================================================
    # RESOLVE DOCLING REFERENCE
    # ========================================================

    def resolve_ref(
        self,
        root: dict,
        ref: str
    ):

        if not ref:

            return None

        if not ref.startswith("#/"):

            return None

        current = root

        for part in ref[2:].split("/"):

            if isinstance(
                current,
                list
            ):

                try:

                    current = current[
                        int(part)
                    ]

                except (
                    ValueError,
                    IndexError
                ):

                    return None

            elif isinstance(
                current,
                dict
            ):

                current = current.get(
                    part
                )

            else:

                return None

            if current is None:

                return None

        return current

    # ========================================================
    # PAGE EXTRACTION
    # ========================================================

    def get_pages(
        self,
        item: dict
    ):

        pages = []

        for prov in item.get(
            "prov",
            []
        ):

            page = prov.get(
                "page_no"
            )

            if page is not None:

                pages.append(
                    page
                )

        return sorted(
            set(pages)
        )

    # ========================================================
    # ITEM TEXT
    # ========================================================

    def get_item_text(
        self,
        item: dict
    ) -> str:

        return self.clean_text(
            item.get(
                "text",
                ""
            )
        )

    # ========================================================
    # WALK DOCLING TREE
    # ========================================================

    def walk_docling_tree(
        self,
        root,
        children
    ):

        """
        Flatten the complete Docling document tree.

        Groups can contain:

            - text
            - headings
            - pictures
            - tables
        """

        for child in children:

            ref = child.get(
                "$ref"
            )

            if not ref:

                continue

            item = self.resolve_ref(
                root,
                ref
            )

            if not item:

                continue

            label = item.get(
                "label",
                ""
            )

            # ------------------------------------------------
            # Nested group
            # ------------------------------------------------

            if label == "group":

                nested_children = (
                    item.get(
                        "children",
                        []
                    )
                )

                yield from (
                    self.walk_docling_tree(
                        root,
                        nested_children
                    )
                )

            else:

                yield item

    # ========================================================
    # HEADING DETECTION
    # ========================================================

    def is_numbered_heading(
        self,
        title: str
    ) -> bool:

        return bool(
            re.match(
                r"^\d+(?:\.\d+)*\s+",
                title.strip()
            )
        )

    # ========================================================
    # HEADING DEPTH
    # ========================================================

    def heading_depth(
        self,
        title: str
    ) -> int:

        match = re.match(
            r"^(\d+(?:\.\d+)*)\s+",
            title.strip()
        )

        if not match:

            return 1

        return len(
            match.group(1).split(".")
        )

    # ========================================================
    # REFERENCES
    # ========================================================

    def is_references_heading(
        self,
        title: str
    ) -> bool:

        return (
            self.clean_text(
                title
            ).lower()
            in {
                "references",
                "bibliography",
                "references and bibliography"
            }
        )

    # ========================================================
    # APPENDIX
    # ========================================================

    def is_appendix_heading(
        self,
        title: str
    ) -> bool:

        normalized = self.clean_text(
            title
        ).lower()

        return (
            normalized == "appendix"
            or normalized.startswith(
                "appendix "
            )
            or normalized.startswith(
                "appendix:"
            )
        )

    # ========================================================
    # UPDATE SECTION PATH
    # ========================================================

    def update_section_path(
        self,
        current_path,
        title
    ):

        title = self.clean_text(
            title
        )

        if not self.is_numbered_heading(
            title
        ):

            return current_path

        depth = self.heading_depth(
            title
        )

        return (
            current_path[
                :depth - 1
            ]
            + [title]
        )

    # ========================================================
    # DOCUMENT MARKER
    # ========================================================

    def is_document_marker(
        self,
        item: dict,
        predicate
    ) -> bool:

        label = str(
            item.get(
                "label",
                ""
            )
        ).lower()

        if label not in {
            "section_header",
            "text",
            "paragraph",
            "list_item",
        }:

            return False

        return predicate(
            self.get_item_text(
                item
            )
        )

    # ========================================================
    # PARSE ELEMENTS
    # ========================================================

    def _parse_elements(
        self,
        root
    ):

        body = root.get(
            "body",
            {}
        )

        body_children = body.get(
            "children",
            []
        )

        raw_items = list(
            self.walk_docling_tree(
                root,
                body_children
            )
        )

        print(
            f"Flattened Docling items: "
            f"{len(raw_items)}"
        )

        elements = []

        section_path = []

        document_part = "main"

        for item in raw_items:

            label = str(
                item.get(
                    "label",
                    ""
                )
            ).lower()

            title = self.get_item_text(
                item
            )

            # =================================================
            # APPENDIX FALLBACK
            # =================================================

            item_pages = self.get_pages(
                item
            )

            if (
                item_pages
                and min(item_pages) >= 13
                and document_part != "appendix"
            ):

                document_part = "appendix"

                section_path = [
                    "APPENDIX"
                ]

            # =================================================
            # REFERENCES MARKER
            # =================================================

            if self.is_document_marker(
                item,
                self.is_references_heading
            ):

                document_part = "references"

                section_path = []

                self.document_order += 1

                continue

            # =================================================
            # APPENDIX MARKER
            # =================================================

            if self.is_document_marker(
                item,
                self.is_appendix_heading
            ):

                document_part = "appendix"

                section_path = [
                    "APPENDIX"
                ]

                self.document_order += 1

                continue

            # =================================================
            # TEXT
            # =================================================

            if label in {
                "text",
                "paragraph",
                "list_item"
            }:

                # References text is ignored.
                if document_part == "references":

                    self.document_order += 1

                    continue

                if title:

                    elements.append(
                        self._create_element(
                            element_type="text",
                            item=item,
                            text=title,
                            section_path=section_path,
                            document_part=document_part
                        )
                    )

                self.document_order += 1

                continue

            # =================================================
            # SECTION HEADER
            # =================================================

            if label == "section_header":

                if not title:

                    self.document_order += 1

                    continue

                if document_part == "references":

                    self.document_order += 1

                    continue

                if self.is_numbered_heading(
                    title
                ):

                    section_path = (
                        self.update_section_path(
                            section_path,
                            title
                        )
                    )

                elements.append(
                    self._create_element(
                        element_type="heading",
                        item=item,
                        text=title,
                        section_path=section_path,
                        document_part=document_part
                    )
                )

                self.document_order += 1

                continue

            # =================================================
            # TABLE
            # =================================================

            if label == "table":

                elements.append(
                    self._create_element(
                        element_type="table",
                        item=item,
                        section_path=section_path,
                        document_part=document_part
                    )
                )

                self.document_order += 1

                continue

            # =================================================
            # PICTURE
            # =================================================

            if label == "picture":

                elements.append(
                    self._create_element(
                        element_type="picture",
                        item=item,
                        section_path=section_path,
                        document_part=document_part
                    )
                )

                self.document_order += 1

                continue

            self.document_order += 1

        return elements

    # ========================================================
    # CREATE ELEMENT
    # ========================================================

    def _create_element(
        self,
        element_type,
        item,
        section_path,
        document_part,
        text=""
    ):

        pages = self.get_pages(
            item
        )

        return Element(

            element_type=element_type,

            index=self.document_order,

            text=text,

            item=(
                item
                if element_type
                in {
                    "table",
                    "picture"
                }
                else None
            ),

            section_path=(
                section_path.copy()
            ),

            page_start=(
                min(pages)
                if pages
                else None
            ),

            page_end=(
                max(pages)
                if pages
                else None
            ),

            document_part=document_part
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    parser = DocumentParser()

    document_path = Path(
        "data/extracted/attention/document.json"
    )

    elements = parser.parse(
        document_path
    )

    print()
    print("=" * 60)
    print("DOCUMENT PARSER TEST")
    print("=" * 60)

    print(
        f"Total elements: {len(elements)}"
    )

    counts = {}

    for element in elements:

        counts[
            element.element_type
        ] = (
            counts.get(
                element.element_type,
                0
            )
            + 1
        )

    print()

    for element_type, count in (
        counts.items()
    ):

        print(
            f"{element_type:<12}: {count}"
        )

    print()
    print("First 10 elements")
    print("-" * 60)

    for element in elements[:10]:

        print(
            f"Index   : {element.index}"
        )

        print(
            f"Type    : "
            f"{element.element_type}"
        )

        print(
            f"Page    : "
            f"{element.page_start}-"
            f"{element.page_end}"
        )

        print(
            f"Section : "
            f"{' > '.join(element.section_path)}"
        )

        if element.text:

            print(
                f"Text    : "
                f"{element.text[:100]}"
            )

        print()