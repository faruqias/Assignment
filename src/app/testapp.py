from src.app.application import RAGApplication

PDF_PATH = "data/pdfs/attention.pdf"


def main():

    print("=" * 70)
    print("RAG APPLICATION TEST")
    print("=" * 70)

    # ========================================================
    # INITIALIZE APPLICATION
    # ========================================================

    app = RAGApplication()

    # ========================================================
    # TEST DOCUMENT UPLOAD
    # ========================================================

    print()
    print("=" * 70)
    print("TEST 1 - DOCUMENT UPLOAD")
    print("=" * 70)

    result = app.upload_document(
        PDF_PATH
    )

    print()
    print("Upload result:")
    print(result)

    # ========================================================
    # TEST QUESTION
    # ========================================================

    question = (
        "How does scaled dot-product attention work?"
    )

    print()
    print("=" * 70)
    print("TEST 2 - QUESTION")
    print("=" * 70)

    print(
        f"Question: {question}"
    )

    answer = ""

    final_results = []

    print()
    print("Generating answer...")
    print()

    for partial_answer, results in app.ask_question(
        question
    ):

        answer = partial_answer
        final_results = results

        print(
            "\r" + answer,
            end="",
            flush=True
        )

    print()
    print()

    # ========================================================
    # SOURCES
    # ========================================================

    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    sources = app.get_sources(
        final_results
    )

    print(sources)

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if not answer.strip():

        raise RuntimeError(
            "RAGApplication returned an empty answer."
        )

    if not final_results:

        raise RuntimeError(
            "RAGApplication returned no retrieval results."
        )

    print(
        f"Answer length : {len(answer)}"
    )

    print(
        f"Sources       : {len(final_results)}"
    )

    print()
    print(
        "RAG APPLICATION VALIDATION PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()