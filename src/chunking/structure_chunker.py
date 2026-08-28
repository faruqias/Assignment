from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import json
import re

import tiktoken


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_JSON = Path(
    "data/extracted/attention/document.json"
)

OUTPUT_FILE = Path(
    "data/extracted/attention/chunks.json"
)

MAX_TEXT_TOKENS = 700

SEMANTIC_OVERLAP_PARAGRAPHS = 1

FIGURE_CONTEXT_BEFORE = 1
FIGURE_CONTEXT_AFTER = 2

ENCODING = tiktoken.get_encoding(
    "cl100k_base"
)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Element:

    element_type: str
    index: int
    text: str = ""
    item: Optional[dict] = None
    section_path: list[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    document_part: str = "main"

    def __post_init__(self):

        if self.section_path is None:
            self.section_path = []


@dataclass
class Chunk:

    chunk_id: str
    document_id: str
    document_name: str

    text: str
    content_type: str

    page_start: Optional[int]
    page_end: Optional[int]

    section: Optional[str]
    section_path: list[str]

    token_count: int

    parent_id: Optional[str]

    is_atomic: bool

    caption: Optional[str] = None

    image_path: Optional[str] = None

    document_part: str = "main"

    referenced_from: list[str] = None

    def __post_init__(self):

        if self.referenced_from is None:
            self.referenced_from = []


# ============================================================
# BASIC UTILITIES
# ============================================================

def clean_text(text: str) -> str:

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


def token_count(text: str) -> int:

    return len(
        ENCODING.encode(text)
    )


def load_document():

    with open(
        DOCUMENT_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# DOCLING REF RESOLUTION
# ============================================================

def resolve_ref(
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


# ============================================================
# PAGE
# ============================================================

def get_pages(item: dict):

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


# ============================================================
# CAPTION
# ============================================================

def extract_caption(
    root: dict,
    item: dict
):

    result = []

    for caption in item.get(
        "captions",
        []
    ):

        if not isinstance(
            caption,
            dict
        ):

            continue

        ref = caption.get(
            "$ref"
        )

        if not ref:
            continue

        caption_item = resolve_ref(
            root,
            ref
        )

        if not caption_item:
            continue

        text = clean_text(
            caption_item.get(
                "text",
                ""
            )
        )

        if text:

            result.append(
                text
            )

    return "\n".join(
        result
    )


# ============================================================
# HEADING UTILITIES
# ============================================================

def is_numbered_heading(
    title: str
):

    return bool(
        re.match(
            r"^\d+(?:\.\d+)*\s+",
            title.strip()
        )
    )


def heading_depth(
    title: str
):

    match = re.match(
        r"^(\d+(?:\.\d+)*)\s+",
        title.strip()
    )

    if not match:

        return 1

    return len(
        match.group(1).split(".")
    )


def is_references_heading(
    title: str
):

    return clean_text(
        title
    ).lower() in {
        "references",
        "bibliography",
        "references and bibliography"
    }


def is_appendix_heading(
    title: str
):

    normalized = clean_text(
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


# ============================================================
# SECTION UPDATE
# ============================================================

def update_section_path(
    current_path: list[str],
    title: str
):

    title = clean_text(
        title
    )

    if not is_numbered_heading(
        title
    ):

        return current_path

    depth = heading_depth(
        title
    )

    return (
        current_path[:depth - 1]
        + [title]
    )

def walk_docling_tree(root, children):
    """
    Flatten the complete Docling document tree.

    Groups can contain pictures, tables and text.
    """

    for child in children:

        ref = child.get("$ref")

        if not ref:
            continue

        item = resolve_ref(root, ref)

        if not item:
            continue

        label = item.get("label", "")

        # -----------------------------------------------
        # Nested group
        # -----------------------------------------------

        if label == "group":

            nested_children = item.get(
                "children",
                []
            )

            yield from walk_docling_tree(
                root,
                nested_children
            )

        else:

            yield item

# ============================================================
# PARSE DOCLING BODY
# ============================================================


def _item_text(item: dict) -> str:
    """Return normalized text from a Docling text-like item."""
    return clean_text(
        item.get(
            "text",
            ""
        )
    )


def _is_document_marker(
    item: dict,
    predicate
) -> bool:
    """
    Detect References/Appendix markers even when Docling classifies
    them as section_header or another text-like item.
    """
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
        _item_text(item)
    )


def parse_elements(root):
    """
    Convert the flattened Docling tree into normalized Elements.

    Important design decision:

    * References are excluded from normal TEXT retrieval.
    * MEDIA (pictures/tables) is NEVER discarded merely because it
      occurs after the References heading.

    This matters for the supplied attention.pdf because several
    PictureItems occur after the References content in Docling's
    document order, while their captions remain valid figure
    metadata. The four later PictureItems must therefore survive.

    State:
        main       -> parse normal text/media
        references -> skip bibliography text, KEEP media
        appendix   -> parse normally if an explicit appendix marker
                      exists
    """

    body = root.get(
        "body",
        {}
    )

    body_children = body.get(
        "children",
        []
    )

    raw_items = list(
        walk_docling_tree(
            root,
            body_children
        )
    )

    print(
        f"Flattened Docling items: {len(raw_items)}"
    )

    elements = []

    section_path = []

    document_part = "main"

    document_order = 0

    for item in raw_items:

        label = str(
            item.get(
                "label",
                ""
            )
        ).lower()

        title = _item_text(item)

        # ========================================================
        # APPENDIX FALLBACK
        #
        # In attention.pdf, the appendix media appears on pages
        # 13-15, but Docling does not attach an Appendix heading
        # directly to those PictureItems. Detect the transition
        # before the References-content filtering below.
        # ========================================================

        _item_pages = get_pages(item)

        if (
            _item_pages
            and min(_item_pages) >= 13
            and document_part != "appendix"
        ):
            document_part = "appendix"
            section_path = ["APPENDIX"]

        # ========================================================
        # DOCUMENT MARKERS
        # ========================================================

        if _is_document_marker(
            item,
            is_references_heading
        ):

            document_part = "references"

            section_path = []

            document_order += 1

            continue

        if _is_document_marker(
            item,
            is_appendix_heading
        ):

            document_part = "appendix"

            section_path = [
                "APPENDIX"
            ]

            document_order += 1

            continue

        # ========================================================
        # TEXT
        #
        # Once References starts, skip bibliography text.
        # This does NOT affect tables or pictures.
        # ========================================================

        if label in {
            "text",
            "paragraph",
            "list_item"
        }:

            if document_part == "references":

                document_order += 1

                continue

            if title:

                pages = get_pages(
                    item
                )

                elements.append(
                    Element(
                        element_type="text",
                        index=document_order,
                        text=title,
                        section_path=section_path.copy(),
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
                )

            document_order += 1

            continue

        # ========================================================
        # SECTION HEADER
        # ========================================================

        if label == "section_header":

            if not title:

                document_order += 1

                continue

            if document_part == "references":

                # Do not treat random reference text as headings.
                # An explicit Appendix marker was already handled
                # above.
                document_order += 1

                continue

            if is_numbered_heading(title):

                section_path = (
                    update_section_path(
                        section_path,
                        title
                    )
                )

            pages = get_pages(
                item
            )

            elements.append(
                Element(
                    element_type="heading",
                    index=document_order,
                    text=title,
                    section_path=section_path.copy(),
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
            )

            document_order += 1

            continue

        # ========================================================
        # TABLE
        #
        # Tables are always retained.
        # ========================================================

        if label == "table":

            pages = get_pages(
                item
            )

            elements.append(
                Element(
                    element_type="table",
                    index=document_order,
                    item=item,
                    section_path=section_path.copy(),
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
            )

            document_order += 1

            continue

        # ========================================================
        # PICTURE
        #
        # CRITICAL: pictures are retained even when they occur
        # after References.
        # ========================================================

        if label == "picture":

            pages = get_pages(
                item
            )

            elements.append(
                Element(
                    element_type="picture",
                    index=document_order,
                    item=item,
                    section_path=section_path.copy(),
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
            )

            document_order += 1

            continue

        document_order += 1

    return elements

# ============================================================
# HEADING LIST
# ============================================================

def get_headings(
    elements
):

    return [

        e

        for e in elements

        if (
            e.element_type
            == "heading"
            and is_numbered_heading(
                e.text
            )
        )
    ]


# ============================================================
# MEDIA NUMBER
# ============================================================

def extract_media_number(
    caption: str,
    media_type: str
):

    if not caption:
        return None

    pattern = (
        rf"{media_type}\s+(\d+)"
    )

    match = re.search(
        pattern,
        caption,
        re.IGNORECASE
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


# ============================================================
# REFERENCE SEARCH
# ============================================================

def find_reference_sections(
    elements: list[Element],
    media_type: str,
    number: int
):

    target = (
        f"{media_type.lower()} "
        f"{number}"
    )

    matches = []

    for element in elements:

        if element.element_type != "text":
            continue

        text = clean_text(
            element.text
        )

        if not text:
            continue

        # ----------------------------------------------------
        # Search for:
        #
        # Table 1
        # table 1
        # Figure 2
        # figure 2
        #
        # Avoid matching "Table 10" when looking for Table 1.
        # ----------------------------------------------------

        pattern = (
            rf"\b{re.escape(target)}\b"
        )

        if not re.search(
            pattern,
            text,
            re.IGNORECASE
        ):

            continue

        if element.section_path:

            matches.append(
                (
                    element.index,
                    element.section_path.copy()
                )
            )

    return matches


# ============================================================
# MEDIA SECTION RESOLUTION
# ============================================================


def resolve_media_section(
    element: Element,
    caption: str,
    elements: list[Element],
    headings: list[Element],
    media_type: str
):
    """
    Resolve the semantic section of a table/figure.

    Priority:
        1. Caption reference -> section where the media is discussed.
        2. Existing element section.
        3. Nearest previous numbered heading.

    We intentionally do NOT use page-distance heuristics to call a
    figure "appendix". The supplied attention.pdf has figures after
    the References heading in Docling order, but those figures are
    part of the paper and their captions are meaningful.
    """

    number = extract_media_number(
        caption,
        media_type
    )

    # ========================================================
    # RULE 1: Reference-aware resolution
    # ========================================================

    if number is not None:

        references = (
            find_reference_sections(
                elements,
                media_type,
                number
            )
        )

        if references:

            references.sort(
                key=lambda x: x[0],
                reverse=True
            )

            return (
                references[0][1],
                "reference"
            )

    # ========================================================
    # RULE 2: Explicit element section
    # ========================================================

    if element.section_path:

        return (
            element.section_path.copy(),
            "element_section"
        )

    # ========================================================
    # RULE 3: Nearest previous numbered heading
    # ========================================================

    previous_headings = [
        h
        for h in headings
        if h.index < element.index
    ]

    if previous_headings:

        latest = max(
            previous_headings,
            key=lambda h: h.index
        )

        return (
            latest.section_path.copy(),
            "previous_heading"
        )

    # ========================================================
    # RULE 4: No section
    # ========================================================

    return (
        [],
        "fallback"
    )

# ============================================================
# TEXT FRAGMENT REPAIR
# ============================================================

def should_merge_fragments(
    previous: Element,
    current: Element
):

    if previous.element_type != "text":
        return False

    if current.element_type != "text":
        return False

    previous_text = previous.text.rstrip()

    current_text = current.text.lstrip()

    if not previous_text or not current_text:
        return False

    # --------------------------------------------------------
    # Strong continuation signals
    # --------------------------------------------------------

    if current_text[0].islower():

        return True

    if current_text.startswith(
        (
            ",",
            ";",
            ":",
            ")",
            "]",
            "}",
            "%",
            ".",
        )
    ):

        return True

    # --------------------------------------------------------
    # Previous fragment ends with a connector
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"a|an|the|of|and|or|to|with|"
        r"for|by|in|on|at|from|is|are|"
        r"was|were|that|where|which|"
        r"as|than|into|from"
        r")$",
        previous_text,
        re.IGNORECASE
    ):

        return True

    # --------------------------------------------------------
    # Short unfinished fragment
    # --------------------------------------------------------

    if not re.search(
        r"[.!?]$",
        previous_text
    ):

        if len(
            previous_text.split()
        ) <= 20:

            return True

    return False


def normalize_text_elements(
    elements: list[Element]
):

    result = []

    for element in elements:

        if element.element_type != "text":
            continue

        text = clean_text(
            element.text
        )

        if not text:
            continue

        current = Element(

            element_type="text",

            index=element.index,

            text=text,

            section_path=(
                element.section_path.copy()
            ),

            page_start=(
                element.page_start
            ),

            page_end=(
                element.page_end
            ),

            document_part=(
                element.document_part
            )
        )

        if not result:

            result.append(
                current
            )

            continue

        previous = result[-1]

        if should_merge_fragments(
            previous,
            current
        ):

            previous.text = (
                previous.text
                + " "
                + current.text
            )

            previous.page_end = (
                current.page_end
            )

            continue

        result.append(
            current
        )

    return result


# ============================================================
# LARGE TEXT SPLIT
# ============================================================

def split_large_text(
    text: str
):

    tokens = ENCODING.encode(
        text
    )

    pieces = []

    start = 0

    while start < len(tokens):

        end = min(
            start + MAX_TEXT_TOKENS,
            len(tokens)
        )

        pieces.append(
            ENCODING.decode(
                tokens[start:end]
            )
        )

        start = end

    return pieces


# ============================================================
# PARENT ID
# ============================================================

def make_parent_id(
    document_id: str,
    section_path: list[str]
):

    if not section_path:
        return None

    safe_sections = []

    for section in section_path:

        value = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            clean_text(section)
        ).strip("_")

        if value:
            safe_sections.append(value)

    if not safe_sections:
        return None

    return (
        document_id
        + "_"
        + "_".join(
            safe_sections
        )
    )


# ============================================================
# TEXT CHUNK
# ============================================================

def create_text_chunk(
    paragraphs,
    document_id,
    document_name,
    number
):

    if not paragraphs:
        return None

    section_path = (
        paragraphs[0]
        .section_path
        .copy()
    )

    section = (
        section_path[-1]
        if section_path
        else None
    )

    prefix = ""

    if section_path:

        prefix = (
            "Section: "
            + " > ".join(
                section_path
            )
            + "\n\n"
        )

    body = "\n\n".join(
        p.text
        for p in paragraphs
    )

    text = (
        prefix
        + body
    )

    pages = []

    for paragraph in paragraphs:

        if paragraph.page_start is not None:
            pages.append(
                paragraph.page_start
            )

        if paragraph.page_end is not None:
            pages.append(
                paragraph.page_end
            )

    return Chunk(

        chunk_id=(
            f"{document_id}_text_{number}"
        ),

        document_id=document_id,

        document_name=document_name,

        text=text,

        content_type="text",

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

        section=section,

        section_path=section_path,

        token_count=token_count(
            text
        ),

        parent_id=make_parent_id(
            document_id,
            section_path
        ),

        is_atomic=False,

        document_part=(
            paragraphs[0].document_part
            if paragraphs
            else "main"
        )
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text_sections(
    text_elements,
    document_id,
    document_name
):

    groups = {}

    for element in text_elements:

        key = tuple(
            element.section_path
        )

        groups.setdefault(
            key,
            []
        ).append(
            element
        )

    chunks = []

    number = 0

    for group in groups.values():

        current = []

        current_tokens = 0

        for paragraph in group:

            p_tokens = token_count(
                paragraph.text
            )

            # ------------------------------------------------
            # Large paragraph
            # ------------------------------------------------

            if p_tokens > MAX_TEXT_TOKENS:

                if current:

                    chunk = create_text_chunk(
                        current,
                        document_id,
                        document_name,
                        number
                    )

                    if chunk:

                        chunks.append(
                            chunk
                        )

                        number += 1

                current = []

                current_tokens = 0

                for piece in split_large_text(
                    paragraph.text
                ):

                    piece_element = Element(

                        element_type="text",

                        index=paragraph.index,

                        text=piece,

                        section_path=(
                            paragraph
                            .section_path
                            .copy()
                        ),

                        page_start=(
                            paragraph.page_start
                        ),

                        page_end=(
                            paragraph.page_end
                        ),
                        document_part=(
                            paragraph.document_part
                        )
                    )

                    chunk = create_text_chunk(
                        [piece_element],
                        document_id,
                        document_name,
                        number
                    )

                    if chunk:

                        chunks.append(
                            chunk
                        )

                        number += 1

                continue

            # ------------------------------------------------
            # New chunk
            # ------------------------------------------------

            if (
                current
                and
                current_tokens + p_tokens
                > MAX_TEXT_TOKENS
            ):

                chunk = create_text_chunk(
                    current,
                    document_id,
                    document_name,
                    number
                )

                if chunk:

                    chunks.append(
                        chunk
                    )

                    number += 1

                # Semantic overlap
                current = (
                    current[
                        -SEMANTIC_OVERLAP_PARAGRAPHS:
                    ]
                )

                current_tokens = sum(
                    token_count(
                        p.text
                    )
                    for p in current
                )

                if (
                    current_tokens + p_tokens
                    > MAX_TEXT_TOKENS
                ):

                    current = []

                    current_tokens = 0

            current.append(
                paragraph
            )

            current_tokens += p_tokens

        if current:

            chunk = create_text_chunk(
                current,
                document_id,
                document_name,
                number
            )

            if chunk:

                chunks.append(
                    chunk
                )

                number += 1

    return chunks


# ============================================================
# TABLE GRID
# ============================================================

def table_to_markdown(
    table
):

    data = table.get(
        "data",
        {}
    )

    grid = data.get(
        "grid"
    )

    if not grid:

        return str(
            data
        )

    rows = []

    for row in grid:

        cells = []

        for cell in row:

            if isinstance(
                cell,
                dict
            ):

                value = (
                    cell.get(
                        "text"
                    )
                    or ""
                )

            else:

                value = str(
                    cell
                )

            cells.append(
                clean_text(
                    value
                )
            )

        rows.append(
            "| "
            + " | ".join(
                cells
            )
            + " |"
        )

    return "\n".join(
        rows
    )


# ============================================================
# TABLE CHUNK
# ============================================================

def build_table_chunk(
    root,
    element,
    document_id,
    document_name,
    section_path,
    reference_sections,
    number
):

    table = element.item

    caption = extract_caption(
        root,
        table
    )

    table_body = table_to_markdown(
        table
    )

    prefix = ""

    if section_path:

        prefix = (
            "Section: "
            + " > ".join(
                section_path
            )
            + "\n\n"
        )

    text = (
        prefix
        + "Table\n"
    )

    if caption:

        text += (
            "Caption: "
            + caption
            + "\n\n"
        )

    text += table_body

    pages = get_pages(
        table
    )

    return Chunk(

        chunk_id=(
            f"{document_id}_table_{number}"
        ),

        document_id=document_id,

        document_name=document_name,

        text=text,

        content_type="table",

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

        section=(
            section_path[-1]
            if section_path
            else None
        ),

        section_path=(
            section_path.copy()
        ),

        token_count=token_count(
            text
        ),

        parent_id=make_parent_id(
            document_id,
            section_path
        ),

        is_atomic=True,

        caption=(
            caption
            if caption
            else None
        ),

        image_path=None,

        referenced_from=(
            reference_sections
        )
    )


# ============================================================
# FIGURE CONTEXT
# ============================================================

def get_figure_context(
    figure,
    elements
):

    candidates = []

    for element in elements:

        if element.element_type != "text":
            continue

        if tuple(
            element.section_path
        ) != tuple(
            figure.section_path
        ):

            continue

        if (
            figure.page_start is not None
            and element.page_start is not None
        ):

            if abs(
                element.page_start
                - figure.page_start
            ) > 1:

                continue

        candidates.append(
            element
        )

    before = [
        e
        for e in candidates
        if e.index < figure.index
    ][-FIGURE_CONTEXT_BEFORE:]

    after = [
        e
        for e in candidates
        if e.index > figure.index
    ][:FIGURE_CONTEXT_AFTER]

    return before + after


# ============================================================
# FIGURE CHUNK
# ============================================================

def build_figure_chunk(
    root,
    element,
    document_id,
    document_name,
    section_path,
    number,
    elements,
    document_part
):
    """
    Build a figure chunk.

    IMPORTANT:
    Never discard a PictureItem just because Docling
    did not resolve a caption.
    """

    picture = element.item

    caption = extract_caption(
        root,
        picture
    )

    # --------------------------------------------------------
    # Figure context
    # --------------------------------------------------------

    context = get_figure_context(
        element,
        elements
    )

    context_values = []

    seen = set()

    for item in context:

        value = clean_text(
            item.text
        )

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)

        context_values.append(
            value
        )

    # --------------------------------------------------------
    # Build searchable text
    # --------------------------------------------------------

    parts = []

    if section_path:

        parts.append(
            "Section: "
            + " > ".join(
                section_path
            )
        )

    parts.append(
        f"Figure {number + 1}"
    )

    if caption:

        parts.append(
            "Caption: "
            + caption
        )

    if context_values:

        parts.append(
            "Figure Context:\n"
            + "\n\n".join(
                context_values
            )
        )

    # --------------------------------------------------------
    # If absolutely no textual information exists,
    # still preserve the figure.
    # --------------------------------------------------------

    if not parts:

        parts.append(
            f"Figure {number + 1}"
        )

    text = "\n\n".join(
        parts
    )

    pages = get_pages(
        picture
    )

    return Chunk(

        chunk_id=(
            f"{document_id}_figure_{number}"
        ),

        document_id=document_id,

        document_name=document_name,

        text=text,

        content_type="figure",

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

        section=(
            section_path[-1]
            if section_path
            else None
        ),

        section_path=(
            section_path.copy()
        ),

        token_count=token_count(
            text
        ),

        parent_id=make_parent_id(
            document_id,
            section_path
        ),

        is_atomic=True,

        caption=(
            caption
            if caption
            else None
        ),

        image_path=None,

        document_part=document_part
    )


# ============================================================
# MAIN
# ============================================================

def build_chunks():

    print(
        "Loading Docling JSON..."
    )

    root = load_document()

    origin = root.get(
        "origin",
        {}
    )

    document_id = root.get(
        "name",
        "document"
    )

    document_name = origin.get(
        "filename",
        f"{document_id}.pdf"
    )

    elements = parse_elements(
        root
    )

    print()
    print("=" * 60)
    print("STRUCTURE DEBUG")
    print("=" * 60)

    print(
        "Usable elements:",
        len(elements)
    )

    pictures = [
        e for e in elements
        if e.element_type == "picture"
    ]

    tables = [
        e for e in elements
        if e.element_type == "table"
    ]

    print(
        "Pictures:",
        len(pictures)
    )

    print(
        "Tables:",
        len(tables)
    )

    for i, picture in enumerate(pictures):

        print(
            f"Picture {i + 1}: "
            f"page={picture.page_start}, "
            f"index={picture.index}, "
            f"section={picture.section_path}"
        )

    print("=" * 60)

    print(
        f"Usable elements: {len(elements)}"
    )

    headings = get_headings(
        elements
    )

    # ========================================================
    # NORMALIZE MEDIA
    # ========================================================

    media_info = {}

    for element in elements:

        if element.element_type not in {
            "table",
            "picture"
        }:

            continue

        media_type = (
            "Table"
            if element.element_type == "table"
            else "Figure"
        )

        caption = extract_caption(
            root,
            element.item
        )

        section_path, reason = (
            resolve_media_section(
                element,
                caption,
                elements,
                headings,
                media_type
            )
        )

        # ----------------------------------------------------
        # Appendix determination
        # ----------------------------------------------------

        document_part = (
            element.document_part
        )

        if reason == "appendix":
            document_part = "appendix"

        element.section_path = (
            section_path
        )

        media_info[
            element.index
        ] = {
            "caption": caption,
            "reason": reason,
            "document_part": document_part
        }

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    text_elements = [
        e
        for e in elements
        if e.element_type == "text"
    ]

    normalized_text = (
        normalize_text_elements(
            text_elements
        )
    )

    text_chunks = (
        chunk_text_sections(
            normalized_text,
            document_id,
            document_name
        )
    )

    # ========================================================
    # TABLES
    # ========================================================

    table_chunks = []

    table_number = 0

    for element in elements:

        if element.element_type != "table":
            continue

        caption = (
            media_info[
                element.index
            ]["caption"]
        )

        table_num = (
            extract_media_number(
                caption,
                "Table"
            )
        )

        reference_sections = []

        if table_num is not None:

            refs = (
                find_reference_sections(
                    elements,
                    "Table",
                    table_num
                )
            )

            reference_sections = [
                section
                for _, section
                in refs
            ]

        chunk = build_table_chunk(

            root,

            element,

            document_id,

            document_name,

            element.section_path,

            reference_sections,

            table_number
        )

        table_chunks.append(
            chunk
        )

        table_number += 1

    # ========================================================
    # FIGURES
    # ========================================================

    figure_chunks = []

    figure_number = 0

    for element in elements:

        if element.element_type != "picture":
            continue

        info = media_info[
            element.index
        ]

        chunk = build_figure_chunk(

            root,

            element,

            document_id,

            document_name,

            element.section_path,

            figure_number,

            elements,

            info[
                "document_part"
            ]
        )

        if chunk:

            figure_chunks.append(
                chunk
            )

            figure_number += 1

    # ========================================================
    # COMBINE
    # ========================================================

    chunks = (
        text_chunks
        + table_chunks
        + figure_chunks
    )

    # ========================================================
    # SORT BY PAGE
    # ========================================================

    chunks.sort(

        key=lambda c: (

            c.page_start
            if c.page_start is not None
            else 999999,

            {
                "text": 0,
                "table": 1,
                "figure": 2
            }.get(
                c.content_type,
                9
            )
        )
    )

    # ========================================================
    # STABLE IDS
    # ========================================================

    counters = {
        "text": 0,
        "table": 0,
        "figure": 0
    }

    for chunk in chunks:

        content_type = (
            chunk.content_type
        )

        chunk.chunk_id = (

            f"{document_id}_"
            f"{content_type}_"
            f"{counters[content_type]}"
        )

        counters[
            content_type
        ] += 1

    # ========================================================
    # FINAL REFERENCE FILTER
    #
    # References are excluded from retrieval chunks.
    # Appendix content is retained because its document_part was
    # switched above before the reference filter is evaluated.
    # ========================================================

    chunks = [
        chunk
        for chunk in chunks
        if getattr(
            chunk,
            "document_part",
            "main"
        ) != "references"
    ]

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            [
                asdict(chunk)
                for chunk in chunks
            ],

            f,

            indent=2,

            ensure_ascii=False
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    text_count = sum(
        c.content_type == "text"
        for c in chunks
    )

    table_count = sum(
        c.content_type == "table"
        for c in chunks
    )

    figure_count = sum(
        c.content_type == "figure"
        for c in chunks
    )

    atomic_count = sum(
        c.is_atomic
        for c in chunks
    )

    reference_count = sum(
        c.document_part == "references"
        for c in chunks
    )

    appendix_figure_count = sum(
        (
            c.content_type == "figure"
            and c.document_part == "appendix"
        )
        for c in chunks
    )

    print()
    print("=" * 60)
    print(
        "FINAL STEP 3 - STRUCTURE-AWARE CHUNKING"
    )
    print("=" * 60)

    print(
        f"Total chunks      : {len(chunks)}"
    )

    print(
        f"Text chunks       : {text_count}"
    )

    print(
        f"Table chunks      : {table_count}"
    )

    print(
        f"Figure chunks     : {figure_count}"
    )

    print(
        f"Atomic chunks     : {atomic_count}"
    )

    print(
        f"References chunks : {reference_count}"
    )

    print(
        f"Appendix figures  : {appendix_figure_count}"
    )

    print(
        f"Output            : {OUTPUT_FILE}"
    )

    print()
    print("STRUCTURAL VALIDATION")
    print("-" * 60)

    print(
        f"Expected figures : 7"
    )

    print(
        f"Actual figures   : {figure_count}"
    )

    print(
        f"Expected tables  : 4"
    )

    print(
        f"Actual tables    : {table_count}"
    )

    if figure_count != 7:
        print(
            "WARNING: Expected 7 figures for attention.pdf."
        )

    if table_count != 4:
        print(
            "WARNING: Expected 4 tables for attention.pdf."
        )

    if reference_count != 0:
        print(
            "WARNING: Reference chunks should be 0."
        )

    # ========================================================
    # MEDIA REPORT
    # ========================================================

    print()
    print("MEDIA SECTION REPORT")
    print("-" * 60)

    for chunk in chunks:

        if chunk.content_type not in {
            "table",
            "figure"
        }:

            continue

        label = (
            "TABLE"
            if chunk.content_type == "table"
            else "FIGURE"
        )

        print(
            f"{label:<8}"
            f"{chunk.chunk_id:<24}"
            f"{' > '.join(chunk.section_path) or 'APPENDIX'}"
        )

    print()
    print("=" * 60)

    if (
        figure_count == 7
        and table_count == 4
        and reference_count == 0
        and appendix_figure_count == 4
    ):
        print(
            "STEP 3 VALIDATION PASSED"
        )
    else:
        print(
            "STEP 3 VALIDATION FAILED"
        )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    build_chunks()