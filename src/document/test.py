from pathlib import Path

from src.document.document_processor import DocumentProcessor
from src.document.document_parser import DocumentParser
from src.chunking.structure_chunker_new import StructureChunker
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_indexer import VectorIndexer


PDF_PATH = "data/pdfs/attention.pdf"

FAISS_PATH = "data/vectorstore/test_index.faiss"
METADATA_PATH = "data/vectorstore/test_metadata.pkl"


def main():

    print("=" * 70)
    print("VECTOR INDEXER DEBUG")
    print("=" * 70)

    # =========================================================
    # 1. Document Processor
    # =========================================================

    print()
    print("1. Running DocumentProcessor...")

    processor = DocumentProcessor()

    result = processor.process(
        PDF_PATH
    )

    print()
    print(
        "Document:",
        result["document_name"]
    )

    print(
        "Document ID:",
        result["document_id"]
    )

    # =========================================================
    # 2. Document Parser
    # =========================================================

    print()
    print("=" * 70)
    print("2. Running DocumentParser")
    print("=" * 70)

    parser = DocumentParser()

    elements = parser.parse(
        result["json_path"]
    )

    print(
        "Elements:",
        len(elements)
    )

    if not elements:

        raise RuntimeError(
            "DocumentParser returned zero elements."
        )

    # =========================================================
    # 3. Structure Chunker
    # =========================================================

    print()
    print("=" * 70)
    print("3. Running StructureChunker")
    print("=" * 70)

    chunker = StructureChunker()

    chunks = chunker.chunk(
        elements
    )

    print(
        "Chunks:",
        len(chunks)
    )

    if not chunks:

        raise RuntimeError(
            "StructureChunker returned zero chunks."
        )

    # =========================================================
    # 4. Embedding Service
    # =========================================================

    print()
    print("=" * 70)
    print("4. Running EmbeddingService")
    print("=" * 70)

    embedding = EmbeddingService()

    print(
        "Embedding dimension:",
        embedding.dimension
    )

    # =========================================================
    # 5. Generate embeddings
    # =========================================================

    print()
    print("=" * 70)
    print("5. Generating Embeddings")
    print("=" * 70)

    vectors = embedding.embed_documents(
        chunks
    )

    print()
    print(
        "Vector shape:",
        vectors.shape
    )

    print(
        "Vector dtype:",
        vectors.dtype
    )

    # =========================================================
    # 6. Validate embeddings
    # =========================================================

    if len(vectors) != len(chunks):

        raise RuntimeError(
            "Number of vectors does not match number of chunks."
        )

    if vectors.shape[1] != embedding.dimension:

        raise RuntimeError(
            "Vector dimension does not match embedding dimension."
        )

    # =========================================================
    # 7. Create VectorIndexer
    # =========================================================

    print()
    print("=" * 70)
    print("6. Creating VectorIndexer")
    print("=" * 70)

    Path(
        FAISS_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    indexer = VectorIndexer(
        index_path=FAISS_PATH,
        metadata_path=METADATA_PATH
    )

    print(
        "VectorIndexer created."
    )

    print(
        "Dimension:",
        indexer.dimension
    )

    print(
        "Ready:",
        indexer.ready
    )

    # =========================================================
    # 8. Index documents
    # =========================================================

    print()
    print("=" * 70)
    print("7. Indexing Documents")
    print("=" * 70)

    indexer.index_documents(
        chunks,
        vectors
    )

    print()
    print(
        "Indexing completed."
    )

    # =========================================================
    # 9. Inspect FAISS index
    # =========================================================

    print()
    print("=" * 70)
    print("8. FAISS INDEX")
    print("=" * 70)

    print(
        "Vector count:",
        indexer.vector_count
    )

    print(
        "Dimension:",
        indexer.dimension
    )

    print(
        "Expected vectors:",
        len(chunks)
    )

    if indexer.vector_count != len(chunks):

        raise RuntimeError(
            "FAISS vector count does not match chunk count."
        )

    if indexer.dimension != embedding.dimension:

        raise RuntimeError(
            "FAISS dimension does not match embedding dimension."
        )

    # =========================================================
    # 10. Save index
    # =========================================================

    print()
    print("=" * 70)
    print("9. Saving Index")
    print("=" * 70)

    indexer.save()

    print()
    print(
        "Index saved."
    )

    print(
        "FAISS exists:",
        Path(FAISS_PATH).exists()
    )

    print(
        "Metadata exists:",
        Path(METADATA_PATH).exists()
    )

    if not Path(FAISS_PATH).exists():

        raise RuntimeError(
            "FAISS index file was not created."
        )

    if not Path(METADATA_PATH).exists():

        raise RuntimeError(
            "Metadata file was not created."
        )

    # =========================================================
    # 11. Similarity Search
    # =========================================================

    print()
    print("=" * 70)
    print("10. SIMILARITY SEARCH")
    print("=" * 70)

    query = (
        "How does scaled dot-product attention work?"
    )

    print()
    print(
        "Query:",
        query
    )

    query_vector = embedding.embed_query(
        query
    )

    results = indexer.search(
        query_vector,
        top_k=3
    )

    print()
    print(
        "Search results:",
        len(results)
    )

    if not results:

        raise RuntimeError(
            "VectorIndexer returned zero search results."
        )

    for i, result_item in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"RESULT #{i}"
        )

        print(
            "-" * 60
        )

        print(
            result_item
        )

    # =========================================================
    # 12. Reload Index
    # =========================================================

    print()
    print("=" * 70)
    print("11. RELOADING INDEX")
    print("=" * 70)

    reloaded_indexer = VectorIndexer(
        index_path=FAISS_PATH,
        metadata_path=METADATA_PATH
    )

    reloaded_indexer.load()

    print(
        "Reloaded vector count:",
        reloaded_indexer.vector_count
    )

    print(
        "Reloaded dimension:",
        reloaded_indexer.dimension
    )

    print(
        "Reloaded ready:",
        reloaded_indexer.ready
    )

    if reloaded_indexer.vector_count != len(chunks):

        raise RuntimeError(
            "Reloaded index has incorrect vector count."
        )

    if reloaded_indexer.dimension != embedding.dimension:

        raise RuntimeError(
            "Reloaded index has incorrect dimension."
        )

    if not reloaded_indexer.ready:

        raise RuntimeError(
            "Reloaded index is not ready."
        )

    print()
    print(
        "INDEX RELOAD VALIDATION PASSED"
    )

    # =========================================================
    # 13. Search Reloaded Index
    # =========================================================

    print()
    print("=" * 70)
    print("12. SEARCH RELOADED INDEX")
    print("=" * 70)

    reloaded_results = reloaded_indexer.search(
        query_vector,
        top_k=3
    )

    print(
        "Results:",
        len(reloaded_results)
    )

    if not reloaded_results:

        raise RuntimeError(
            "Reloaded VectorIndexer returned zero results."
        )

    # =========================================================
    # 14. Final Validation
    # =========================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        "Chunks             :",
        len(chunks)
    )

    print(
        "Embeddings         :",
        vectors.shape
    )

    print(
        "FAISS vectors      :",
        indexer.vector_count
    )

    print(
        "FAISS dimension    :",
        indexer.dimension
    )

    print(
        "Search results     :",
        len(results)
    )

    print(
        "Reloaded results   :",
        len(reloaded_results)
    )

    print(
        "FAISS index        :",
        FAISS_PATH
    )

    print(
        "Metadata           :",
        METADATA_PATH
    )

    print()
    print(
        "VECTOR INDEXER VALIDATION PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()