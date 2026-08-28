from pathlib import Path
import json

from docling.document_converter import DocumentConverter


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PDF_DIR = Path("data/pdfs")
OUTPUT_DIR = Path("data/extracted")

def inspect_pictures(document):

    print("\n========== PICTURES ==========")

    picture_count = 0

    for item, level in document.iterate_items():

        if item.__class__.__name__ != "PictureItem":
            continue

        picture_count += 1

        print(f"\nPicture #{picture_count}")
        print(f"Type  : {type(item)}")
        print(f"Level : {level}")

        print("\nLabel:")
        print(item.label)

        print("\nCaptions:")
        print(item.captions)

        print("\nReferences:")
        print(item.references)

        print("\nProvenance:")
        print(item.prov)

        print("\nImage:")
        print(item.image)

        print("\nMetadata:")
        print(item.meta)

    print(f"\nTotal PictureItems: {picture_count}")

def save_images(document, output_dir: Path):

    images_dir = output_dir / "images"

    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    picture_count = 0
    saved_count = 0

    for item, _level in document.iterate_items():

        if item.__class__.__name__ != "PictureItem":
            continue

        picture_count += 1

        image = item.get_image(document)

        if image is None:

            print(
                f"Image unavailable for Picture #{picture_count}"
            )

            continue

        image_path = (
            images_dir /
            f"figure_{picture_count:02d}.png"
        )

        image.save(image_path)

        saved_count += 1

        print(
            f"Image saved: {image_path}"
        )

    print(
        f"\nPictureItems found : {picture_count}"
    )

    print(
        f"Images saved       : {saved_count}"
    )


# ---------------------------------------------------------
# Parser
# ---------------------------------------------------------

def parse_pdf(pdf_path: Path):

    print(f"\nParsing: {pdf_path.name}")

    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    return result.document


# ---------------------------------------------------------
# Save Markdown
# ---------------------------------------------------------

def save_markdown(document, output_path: Path):

    markdown = document.export_to_markdown()

    output_path.write_text(
        markdown,
        encoding="utf-8"
    )

    print(f"Markdown saved: {output_path}")


# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------

def save_json(document, output_path: Path):

    json_data = document.export_to_dict()

    output_path.write_text(
        json.dumps(
            json_data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"JSON saved: {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    pdf_path = PDF_DIR / "attention.pdf"

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Create output directory
    document_name = pdf_path.stem

    output_dir = OUTPUT_DIR / document_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Parse PDF
    document = parse_pdf(pdf_path)

    # Save Markdown
    save_markdown(
        document,
        output_dir / "document.md"
    )

    # Save JSON
    save_json(
        document,
        output_dir / "document.json"
    )

    inspect_pictures(document)


    print("\nExtraction completed successfully.")


if __name__ == "__main__":
    main()