from pathlib import Path
import hashlib
import json


class DocumentRegistry:
    """
    Keeps track of documents that have already been indexed.

    Uses SHA-256 of the PDF contents as document_id.
    """

    def __init__(
        self,
        registry_path="data/vectorstore/documents.json"
    ):

        self.registry_path = Path(
            registry_path
        )

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.documents = self._load()

    # ========================================================
    # LOAD
    # ========================================================

    def _load(self):

        if not self.registry_path.exists():

            return []

        try:

            with open(
                self.registry_path,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

        except (json.JSONDecodeError, OSError):

            return []

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    def calculate_document_id(
        self,
        pdf_path
    ):
        """
        Calculate SHA-256 hash of the PDF.

        The same file produces the same document_id.
        A modified file produces a different document_id.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        sha256 = hashlib.sha256()

        with open(
            pdf_path,
            "rb"
        ) as f:

            while True:

                data = f.read(
                    1024 * 1024
                )

                if not data:

                    break

                sha256.update(
                    data
                )

        return sha256.hexdigest()

    # ========================================================
    # EXISTS
    # ========================================================

    def document_exists(
        self,
        document_id
    ):
        """
        Check whether a document has already
        been registered.
        """

        return any(
            document.get("document_id")
            == document_id
            for document in self.documents
        )

    # ========================================================
    # GET DOCUMENT
    # ========================================================

    def get_document(
        self,
        document_id
    ):

        for document in self.documents:

            if (
                document.get(
                    "document_id"
                )
                == document_id
            ):

                return document

        return None

    # ========================================================
    # REGISTER
    # ========================================================

    def register_document(
        self,
        document_id,
        document_name,
        chunk_count=0
    ):
        """
        Register a successfully indexed document.
        """

        if self.document_exists(
            document_id
        ):

            return False

        document = {
            "document_id": document_id,
            "document_name": document_name,
            "chunk_count": chunk_count
        }

        self.documents.append(
            document
        )

        self._save()

        return True

    # ========================================================
    # REMOVE
    # ========================================================

    def remove_document(
        self,
        document_id
    ):

        original_count = len(
            self.documents
        )

        self.documents = [
            document
            for document in self.documents
            if document.get(
                "document_id"
            ) != document_id
        ]

        removed = (
            len(self.documents)
            != original_count
        )

        if removed:

            self._save()

        return removed

    # ========================================================
    # SAVE
    # ========================================================

    def _save(self):

        with open(
            self.registry_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.documents,
                f,
                indent=2,
                ensure_ascii=False
            )