from dataclasses import dataclass, field
from typing import Any, List, Optional
import json
from pathlib import Path


# ============================================================
# NORMALIZED DOCUMENT ELEMENT
# ============================================================

@dataclass
class Element:

    element_type: str

    text: str = ""

    level: int = 0

    ref: Optional[str] = None

    page_start: Optional[int] = None

    page_end: Optional[int] = None

    section_path: List[str] = field(
        default_factory=list
    )

    caption: Optional[str] = None

    metadata: dict = field(
        default_factory=dict
    )


# ============================================================
# DOCUMENT PARSER
# ============================================================

class DocumentParser:
    """
    Converts a live Docling Document into a list
    of normalized Element objects.

    Input:

        Docling Document

    Output:

        List[Element]
    """

    def __init__(self):

        print(
            "DocumentParser initialized."
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def parse(self, document) -> List[Element]:

        if document is None:
            raise ValueError(
                "Document cannot be None."
            )

        print()
        print("=" * 70)
        print("DOCUMENT PARSER")
        print("=" * 70)

        # --------------------------------------------------------
        # JSON file path
        # --------------------------------------------------------

        if isinstance(
            document,
            (str, Path)
        ):

            print(
                f"Loading JSON: {document}"
            )

            path = Path(document)

            if not path.exists():
                raise FileNotFoundError(
                    f"Document JSON not found: {path}"
                )

            with path.open(
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            elements = self._parse_json(
                data
            )

        # --------------------------------------------------------
        # JSON dictionary
        # --------------------------------------------------------

        elif isinstance(
            document,
            dict
        ):

            print(
                "Parsing JSON dictionary..."
            )

            elements = self._parse_json(
                document
            )

        # --------------------------------------------------------
        # Live Docling Document
        # --------------------------------------------------------

        elif hasattr(
            document,
            "iterate_items"
        ):

            print(
                "Parsing live Docling document..."
            )

            elements = (
                self._parse_docling_document(
                    document
                )
            )

        # --------------------------------------------------------
        # Element list
        # --------------------------------------------------------

        elif isinstance(
            document,
            list
        ):

            print(
                "Parsing element list..."
            )

            elements = (
                self._parse_element_list(
                    document
                )
            )

        else:

            raise TypeError(
                "Unsupported document type: "
                f"{type(document)}"
            )

        print()
        print(
            f"Parsed elements: {len(elements)}"
        )

        self._print_summary(
            elements
        )

        return elements

    # ========================================================
    # DOCling DOCUMENT
    # ========================================================

    def _parse_docling_document(
        self,
        document
    ) -> List[Element]:

        elements = []

        section_stack = []

        for item, level in document.iterate_items():

            item_type = (
                item.__class__.__name__
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            text = getattr(
                item,
                "text",
                ""
            )

            if text is None:
                text = ""

            text = str(text).strip()

            # ------------------------------------------------
            # Reference
            # ------------------------------------------------

            ref = getattr(
                item,
                "self_ref",
                None
            )

            if ref is not None:
                ref = str(ref)

            # ------------------------------------------------
            # Page information
            # ------------------------------------------------

            page_start, page_end = (
                self._get_page_range(
                    item
                )
            )

            # ------------------------------------------------
            # Caption
            # ------------------------------------------------

            caption = (
                self._get_caption(
                    item
                )
            )

            # ------------------------------------------------
            # Section handling
            # ------------------------------------------------

            section_path = list(
                section_stack
            )

            if self._is_heading(
                item_type
            ):

                if text:

                    # Remove deeper sections
                    section_stack = (
                        section_stack[:level]
                    )

                    section_stack.append(
                        text
                    )

                    section_path = list(
                        section_stack
                    )

            # ------------------------------------------------
            # Ignore empty structural items
            # ------------------------------------------------

            if (
                not text
                and not caption
                and not self._is_picture(
                    item_type
                )
                and not self._is_table(
                    item_type
                )
            ):

                continue

            # ------------------------------------------------
            # Create normalized element
            # ------------------------------------------------

            element = Element(

                element_type=item_type,

                text=text,

                level=level,

                ref=ref,

                page_start=page_start,

                page_end=page_end,

                section_path=section_path,

                caption=caption,

                metadata={
                    "docling_type": item_type
                }
            )

            elements.append(
                element
            )

        return elements

    # ========================================================
    # ELEMENT LIST
    # ========================================================

    def _parse_element_list(
        self,
        document
    ) -> List[Element]:

        elements = []

        for item in document:

            if isinstance(
                item,
                Element
            ):

                elements.append(
                    item
                )

                continue

            if not isinstance(
                item,
                dict
            ):

                continue

            element = Element(

                element_type=item.get(
                    "element_type",
                    item.get(
                        "type",
                        "Unknown"
                    )
                ),

                text=item.get(
                    "text",
                    ""
                ),

                level=item.get(
                    "level",
                    0
                ),

                ref=item.get(
                    "ref"
                ),

                page_start=item.get(
                    "page_start"
                ),

                page_end=item.get(
                    "page_end"
                ),

                section_path=item.get(
                    "section_path",
                    []
                ),

                caption=item.get(
                    "caption"
                ),

                metadata=item.get(
                    "metadata",
                    {}
                )
            )

            elements.append(
                element
            )

        return elements

    # ========================================================
    # PAGE RANGE
    # ========================================================

    def _get_page_range(
        self,
        item
    ):

        prov = getattr(
            item,
            "prov",
            None
        )

        if not prov:

            return None, None

        pages = []

        try:

            for provenance in prov:

                page_no = getattr(
                    provenance,
                    "page_no",
                    None
                )

                if page_no is not None:

                    pages.append(
                        int(page_no)
                    )

        except Exception:

            return None, None

        if not pages:

            return None, None

        return (
            min(pages),
            max(pages)
        )

    # ========================================================
    # CAPTION
    # ========================================================

    def _get_caption(
        self,
        item
    ):

        captions = getattr(
            item,
            "captions",
            None
        )

        if not captions:

            return None

        values = []

        try:

            for caption in captions:

                text = getattr(
                    caption,
                    "text",
                    None
                )

                if text:

                    values.append(
                        str(text).strip()
                    )

                elif isinstance(
                    caption,
                    str
                ):

                    values.append(
                        caption.strip()
                    )

        except Exception:

            return None

        if not values:

            return None

        return " ".join(
            values
        )

    def _resolve_ref(
        self,
        root,
        ref
    ):

        if not ref.startswith(
            "#/"
        ):
            return None

        parts = ref[2:].split(
            "/"
        )

        if len(parts) != 2:
            return None

        collection = parts[0]

        try:
            index = int(
                parts[1]
            )
        except ValueError:
            return None

        values = root.get(
            collection,
            []
        )

        if (
            index < 0
            or index >= len(values)
        ):
            return None

        return values[index]

        # ========================================================
    # JSON ITEM -> NORMALIZED ELEMENT
    # ========================================================

    def _json_item_to_element(
        self,
        root,
        item,
        ref,
        level,
        section_stack
    ):

        if not isinstance(item, dict):
            return None

        # ----------------------------------------------------
        # Determine element type
        # ----------------------------------------------------

        if ref.startswith("#/texts/"):
            item_type = item.get(
                "label",
                "TextItem"
            )

        elif ref.startswith("#/tables/"):
            item_type = "TableItem"

        elif ref.startswith("#/pictures/"):
            item_type = "PictureItem"

        else:
            item_type = item.get(
                "label",
                "Unknown"
            )

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        text = item.get(
            "text",
            ""
        )

        if text is None:
            text = ""

        text = str(text).strip()

        # ----------------------------------------------------
        # Extract page information
        # ----------------------------------------------------

        page_start = None
        page_end = None

        prov = item.get(
            "prov",
            []
        )

        pages = []

        if isinstance(prov, list):

            for provenance in prov:

                if not isinstance(
                    provenance,
                    dict
                ):
                    continue

                page_no = provenance.get(
                    "page_no"
                )

                if page_no is not None:

                    try:
                        pages.append(
                            int(page_no)
                        )
                    except (
                        TypeError,
                        ValueError
                    ):
                        pass

        if pages:

            page_start = min(pages)
            page_end = max(pages)

        # ----------------------------------------------------
        # Caption
        # ----------------------------------------------------

        caption = None

        captions = item.get(
            "captions"
        )

        if captions:

            caption_values = []

            for caption_item in captions:

                if isinstance(
                    caption_item,
                    str
                ):

                    caption_values.append(
                        caption_item.strip()
                    )

                elif isinstance(
                    caption_item,
                    dict
                ):

                    caption_text = caption_item.get(
                        "text",
                        ""
                    )

                    if caption_text:

                        caption_values.append(
                            str(
                                caption_text
                            ).strip()
                        )

            if caption_values:

                caption = " ".join(
                    caption_values
                )

        # ----------------------------------------------------
        # Heading / section handling
        # ----------------------------------------------------

        current_section = list(
            section_stack
        )

        is_heading = (
            item.get("label")
            in {
                "section_header",
                "title",
                "heading",
            }
        )

        if is_heading and text:

            if level < len(
                section_stack
            ):

                section_stack[:] = (
                    section_stack[:level]
                )

            section_stack.append(
                text
            )

            current_section = list(
                section_stack
            )

        # ----------------------------------------------------
        # Ignore empty items
        # ----------------------------------------------------

        if (
            not text
            and not caption
            and item_type != "PictureItem"
            and item_type != "TableItem"
        ):

            return None

        # ----------------------------------------------------
        # Create normalized Element
        # ----------------------------------------------------

        return Element(

            element_type=item_type,

            text=text,

            level=level,

            ref=ref,

            page_start=page_start,

            page_end=page_end,

            section_path=current_section,

            caption=caption,

            metadata={
                "docling_type": item_type,
                "json_ref": ref,
                "label": item.get(
                    "label"
                )
            }
        )

    def _parse_json(self, root) -> List[Element]:

        if not isinstance(root, dict):
            raise TypeError(
                "Expected JSON dictionary."
            )

        body = root.get(
            "body",
            {}
        )

        children = body.get(
            "children",
            []
        )

        print(
            f"Docling tree children: "
            f"{len(children)}"
        )

        elements = []

        section_stack = []

        def walk(
            nodes,
            level=0
        ):

            for node in nodes:

                if not isinstance(
                    node,
                    dict
                ):
                    continue

                ref = node.get(
                    "$ref"
                )

                if not ref:
                    continue

                item = self._resolve_ref(
                    root,
                    ref
                )

                if not item:
                    continue

                element = self._json_item_to_element(
                    root,
                    item,
                    ref,
                    level,
                    section_stack
                )

                if element:

                    elements.append(
                        element
                    )

                # ----------------------------------------------
                # Process children recursively
                # ----------------------------------------------

                child_nodes = item.get(
                    "children",
                    []
                )

                if child_nodes:

                    walk(
                        child_nodes,
                        level + 1
                    )

        walk(
            children
        )

        return elements

    # ========================================================
    # TYPE HELPERS
    # ========================================================

    @staticmethod
    def _is_heading(
        item_type
    ):

        return item_type in {
            "SectionHeaderItem",
            "TitleItem",
            "HeadingItem",
        }

    @staticmethod
    def _is_picture(
        item_type
    ):

        return item_type in {
            "PictureItem",
        }

    @staticmethod
    def _is_table(
        item_type
    ):

        return item_type in {
            "TableItem",
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    def _print_summary(
        self,
        elements
    ):

        if not elements:

            print(
                "No document elements found."
            )

            return

        counts = {}

        for element in elements:

            key = element.element_type

            counts[key] = (
                counts.get(
                    key,
                    0
                )
                + 1
            )

        print()
        print(
            "Element distribution:"
        )

        for element_type, count in sorted(
            counts.items()
        ):

            print(
                f"   {element_type:<30} "
                f"{count}"
            )

        pages = [
            element.page_start
            for element in elements
            if element.page_start is not None
        ]

        if pages:

            print()
            print(
                f"Pages detected: "
                f"{min(pages)} - {max(pages)}"
            )