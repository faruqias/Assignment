from pathlib import Path

from docling.document_converter import DocumentConverter


PDF_PATH = Path("data/pdfs/attention.pdf")


def inspect_document():

    converter = DocumentConverter()

    result = converter.convert(PDF_PATH)

    document = result.document

    print("\n========== DOCUMENT STRUCTURE ==========\n")

    for item, level in document.iterate_items():

        item_type = item.__class__.__name__

        print(
            f"Level={level:<2} "
            f"Type={item_type:<25} "
            f"Ref={item.self_ref}"
        )

        # Text
        if hasattr(item, "text") and item.text:

            text = item.text.replace("\n", " ")

            print(
                f"  Text: {text[:150]}"
            )

        # Captions
        if hasattr(item, "captions") and item.captions:

            print(
                f"  Captions: {item.captions}"
            )

        print()


if __name__ == "__main__":
    inspect_document()