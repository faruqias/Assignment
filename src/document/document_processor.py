from pathlib import Path
import json

from docling.document_converter import DocumentConverter


class DocumentProcessor:

    """
    Responsible for loading and processing PDF documents
    using Docling.

    Responsibilities:
        PDF
          ↓
        Docling
          ↓
        Docling Document
          ↓
        document.json
        document.md
        images/
    """

    def __init__(self, output_dir="data/extracted"):

        self.output_dir = Path(output_dir)

        self.converter = DocumentConverter()

    # ========================================================
    # PROCESS PDF
    # ========================================================

    def process(self, pdf_path):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        document_name = pdf_path.stem

        output_dir = (
            self.output_dir
            / document_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print("=" * 60)
        print("DOCUMENT PROCESSOR")
        print("=" * 60)

        print(
            f"PDF     : {pdf_path}"
        )

        print(
            f"Output  : {output_dir}"
        )

        # ----------------------------------------------------
        # Docling
        # ----------------------------------------------------

        print()
        print("Running Docling...")

        result = self.converter.convert(
            pdf_path
        )

        document = result.document

        print(
            "Docling processing completed."
        )

        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        json_path = (
            output_dir
            / "document.json"
        )

        self._save_json(
            document,
            json_path
        )

        # ----------------------------------------------------
        # Save Markdown
        # ----------------------------------------------------

        markdown_path = (
            output_dir
            / "document.md"
        )

        self._save_markdown(
            document,
            markdown_path
        )

        # ----------------------------------------------------
        # Save images
        # ----------------------------------------------------

        images_dir = (
            output_dir
            / "images"
        )

        image_count = self._save_images(
            document,
            images_dir
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        print()
        print(
            f"Document JSON : {json_path}"
        )

        print(
            f"Markdown      : {markdown_path}"
        )

        print(
            f"Images        : {image_count}"
        )

        print("=" * 60)

        return {
            "document": document,
            "document_id": document_name,
            "document_name": pdf_path.name,
            "output_dir": output_dir,
            "json_path": json_path,
            "markdown_path": markdown_path,
            "images_dir": images_dir,
            "image_count": image_count
        }

    # ========================================================
    # SAVE JSON
    # ========================================================

    def _save_json(
        self,
        document,
        output_path
    ):

        data = document.export_to_dict()

        output_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(
            f"JSON saved: {output_path}"
        )

    # ========================================================
    # SAVE MARKDOWN
    # ========================================================

    def _save_markdown(
        self,
        document,
        output_path
    ):

        markdown = (
            document.export_to_markdown()
        )

        output_path.write_text(
            markdown,
            encoding="utf-8"
        )

        print(
            f"Markdown saved: {output_path}"
        )

    # ========================================================
    # SAVE IMAGES
    # ========================================================

    def _save_images(
        self,
        document,
        output_dir
    ):

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        picture_count = 0
        saved_count = 0

        for item, _level in (
            document.iterate_items()
        ):

            if (
                item.__class__.__name__
                != "PictureItem"
            ):

                continue

            picture_count += 1

            image = item.get_image(
                document
            )

            if image is None:

                print(
                    f"Image unavailable "
                    f"for Picture #{picture_count}"
                )

                continue

            image_path = (
                output_dir
                / f"figure_{picture_count:02d}.png"
            )

            image.save(
                image_path
            )

            saved_count += 1

            print(
                f"Image saved: {image_path}"
            )

        print()
        print(
            f"Pictures found : {picture_count}"
        )

        print(
            f"Images saved   : {saved_count}"
        )

        return saved_count