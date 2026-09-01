import json
import os
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import (
    AzureOpenAI,
    AsyncAzureOpenAI,
)

from langchain_core.outputs import (
    Generation,
    LLMResult,
)
from langchain_core.prompt_values import PromptValue

from ragas import (
    EvaluationDataset,
    SingleTurnSample,
    evaluate,
)

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms.base import BaseRagasLLM
from ragas.embeddings import OpenAIEmbeddings

from src.app.azure_openai_client import AzureOpenAIClient
from src.app.prompt_builder import PromptBuilder
from src.app.rag_chatbot import RAGChatbot

from src.document.embedding_service import EmbeddingService
from src.document.vector_indexer import VectorIndexer

from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.rrf_fusion import RRFFusion
from src.retriever.retriever import Retriever
from src.retriever.reranker import BGEReranker


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

FAISS_PATH = (
    "data/vectorstore/index.faiss"
)

METADATA_PATH = (
    "data/vectorstore/metadata.json"
)

OUTPUT_PATH = (
    "data/evaluation/ragas_results.json"
)


# ============================================================
# EVALUATION DATASET
# ============================================================

EVALUATION_DATASET = [

    {
        "question": "What is BERT?",

        "ground_truth": (
            "BERT stands for Bidirectional Encoder "
            "Representations from Transformers. It is a "
            "language representation model that uses "
            "deep bidirectional Transformer representations."
        ),
    },

    {
        "question": (
            "What is BERT's pre-training objective?"
        ),

        "ground_truth": (
            "BERT uses masked language modeling as a "
            "pre-training objective, where selected input "
            "tokens are masked and the model predicts the "
            "original tokens."
        ),
    },

    {
        "question": (
            "What are the main contributions of BERT?"
        ),

        "ground_truth": (
            "BERT introduces deep bidirectional pre-training "
            "and demonstrates that a single pre-trained model "
            "can be fine-tuned for a wide range of NLP tasks."
        ),
    },

    {
        "question": (
            "What datasets were used to evaluate BERT?"
        ),

        "ground_truth": (
            "BERT was evaluated on NLP benchmarks including "
            "GLUE, MultiNLI, and SQuAD."
        ),
    },

    {
        "question": (
            "What experimental results support "
            "BERT's effectiveness?"
        ),

        "ground_truth": (
            "The experiments show that BERT achieves strong "
            "results across multiple NLP benchmarks and "
            "establishes state-of-the-art results on several "
            "language understanding tasks."
        ),
    },
]

class RAGASEmbeddingAdapter:
    """
    Adapter that provides the embed_query() interface
    expected by RAGAS metrics.

    RAGAS 0.4.3 OpenAIEmbeddings exposes embed_text(),
    while ResponseRelevancy expects embed_query().
    """

    def __init__(self, embedding):

        self.embedding = embedding

    def embed_query(
        self,
        text: str,
    ):

        return self.embedding.embed_text(
            text
        )

    def embed_documents(
        self,
        texts
    ):

        return [
            self.embedding.embed_text(
                text
            )
            for text in texts
        ]


# ============================================================
# AZURE RAGAS LLM
# ============================================================

class AzureRAGASLLM(BaseRagasLLM):
    """
    RAGAS 0.4.3 LLM adapter for Azure OpenAI.

    Designed for models such as:

        gpt-5.4-mini

    which require:

        max_completion_tokens

    instead of:

        max_tokens
    """

    def __init__(
        self,
        client: AzureOpenAI,
        async_client: AsyncAzureOpenAI,
        deployment: str,
        max_completion_tokens: int = 500,
    ):

        super().__init__()

        self.client = client

        self.async_client = (
            async_client
        )

        self.deployment = deployment

        self.max_completion_tokens = (
            max_completion_tokens
        )

    # ========================================================
    # GENERATE TEXT
    # ========================================================

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:

        prompt_text = (
            prompt.to_string()
        )

        response = (
            self.client.chat.completions.create(
                model=self.deployment,

                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],

                temperature=temperature,

                max_completion_tokens=(
                    self.max_completion_tokens
                ),

                n=n,

                stop=stop,
            )
        )

        generations = []

        for choice in response.choices:

            generations.append(
                Generation(
                    text=(
                        choice.message.content
                        or ""
                    )
                )
            )

        return LLMResult(
            generations=[
                generations
            ]
        )

    # ========================================================
    # ASYNC GENERATE TEXT
    # ========================================================

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:

        prompt_text = (
            prompt.to_string()
        )

        if temperature is None:

            temperature = 0.01

        response = (
            await self.async_client
            .chat
            .completions
            .create(

                model=self.deployment,

                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],

                temperature=temperature,

                max_completion_tokens=(
                    self.max_completion_tokens
                ),

                n=n,

                stop=stop,
            )
        )

        generations = []

        for choice in response.choices:

            generations.append(
                Generation(
                    text=(
                        choice.message.content
                        or ""
                    )
                )
            )

        return LLMResult(
            generations=[
                generations
            ]
        )

    # ========================================================
    # IS FINISHED
    # ========================================================

    def is_finished(
        self,
        response: LLMResult,
    ) -> bool:

        if response is None:

            return False

        if not response.generations:

            return False

        for group in response.generations:

            for generation in group:

                if generation.text:

                    return True

        return False


# ============================================================
# RAGAS EVALUATOR
# ============================================================

class RAGASEvaluator:

    def __init__(self):

        print()
        print("=" * 70)
        print("INITIALIZING RAGAS EVALUATOR")
        print("=" * 70)

        # =====================================================
        # 1. VECTOR STORE
        # =====================================================

        print()
        print("1. Loading existing vector store...")

        self.indexer = VectorIndexer(
            index_path=FAISS_PATH,
            metadata_path=METADATA_PATH,
        )

        self.indexer.load()

        print(
            f"   Vectors: "
            f"{self.indexer.vector_count}"
        )

        if self.indexer.vector_count == 0:

            raise RuntimeError(
                "FAISS index contains no vectors."
            )

        # =====================================================
        # 2. CHUNKS
        # =====================================================

        print()
        print("2. Loading existing chunks...")

        self.chunks = (
            self.indexer.metadata
        )

        print(
            f"   Chunks: "
            f"{len(self.chunks)}"
        )

        if not self.chunks:

            raise RuntimeError(
                "No chunk metadata found."
            )

        # =====================================================
        # 3. QUERY EMBEDDING SERVICE
        # =====================================================

        print()
        print(
            "3. Loading query embedding service..."
        )

        # IMPORTANT:
        #
        # Documents are NOT embedded again.
        #
        # This service is only used to convert
        # evaluation questions into vectors.

        self.embedding = (
            EmbeddingService()
        )

        # =====================================================
        # 4. BM25
        # =====================================================

        print()
        print("4. Creating BM25...")

        self.bm25 = BM25Retriever(
            self.chunks
        )

        # =====================================================
        # 5. RRF
        # =====================================================

        print()
        print("5. Creating RRF...")

        self.rrf = RRFFusion()

        # =====================================================
        # 6. RETRIEVER
        # =====================================================

        print()
        print("6. Creating Retriever...")

        self.retriever = Retriever(
            indexer=self.indexer,
            embedding_service=self.embedding,
            bm25_retriever=self.bm25,
            rrf_fusion=self.rrf,
        )

        # =====================================================
        # 7. RERANKER
        # =====================================================

        print()
        print("7. Creating BGE Reranker...")

        self.reranker = BGEReranker()

        # =====================================================
        # 8. PROMPT BUILDER
        # =====================================================

        print()
        print("8. Creating Prompt Builder...")

        self.prompt_builder = (
            PromptBuilder()
        )

        # =====================================================
        # 9. AZURE OPENAI
        # =====================================================

        print()
        print(
            "9. Creating Azure OpenAI client..."
        )

        self.openapi_client = (
            AzureOpenAIClient()
        )

        # =====================================================
        # 10. RAG CHATBOT
        # =====================================================

        print()
        print("10. Creating RAG Chatbot...")

        self.chatbot = RAGChatbot(
            retriever=self.retriever,
            reranker=self.reranker,
            prompt_builder=self.prompt_builder,
            openapi_client=self.openapi_client,
            chunks=self.chunks,
        )

        print()
        print("=" * 70)
        print("RAG PIPELINE READY")
        print("=" * 70)

    # =========================================================
    # AZURE SYNC CLIENT
    # =========================================================

    def create_azure_client(self):

        api_key = os.getenv(
            "AZURE_OPENAI_API_KEY"
        )

        endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )

        api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION"
        )

        if not api_key:

            raise RuntimeError(
                "AZURE_OPENAI_API_KEY "
                "is not configured."
            )

        if not endpoint:

            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT "
                "is not configured."
            )

        if not api_version:

            raise RuntimeError(
                "AZURE_OPENAI_API_VERSION "
                "is not configured."
            )

        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    # =========================================================
    # AZURE ASYNC CLIENT
    # =========================================================

    def create_async_azure_client(self):

        api_key = os.getenv(
            "AZURE_OPENAI_API_KEY"
        )

        endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )

        api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION"
        )

        if not api_key:

            raise RuntimeError(
                "AZURE_OPENAI_API_KEY "
                "is not configured."
            )

        if not endpoint:

            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT "
                "is not configured."
            )

        if not api_version:

            raise RuntimeError(
                "AZURE_OPENAI_API_VERSION "
                "is not configured."
            )

        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    # =========================================================
    # RAGAS LLM
    # =========================================================

    def create_evaluator_llm(self):

        print()
        print(
            "Creating RAGAS evaluation LLM..."
        )

        deployment = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT"
        )

        if not deployment:

            raise RuntimeError(
                "AZURE_OPENAI_CHAT_DEPLOYMENT "
                "is not configured."
            )

        client = (
            self.create_azure_client()
        )   

        async_client = (
            self.create_async_azure_client()
        )

        print(
            f"RAGAS LLM: {deployment}"
        )

        return AzureRAGASLLM(
            client=client,
            async_client=async_client,
            deployment=deployment,
            max_completion_tokens=500,
        )

    # =========================================================
    # RAGAS EMBEDDINGS
    # =========================================================

    def create_evaluator_embeddings(self):

        print()
        print(
            "Creating RAGAS evaluation embeddings..."
        )

        deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )

        if not deployment:

            raise RuntimeError(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT "
                "is not configured."
            )

        print(
            f"RAGAS Embedding Deployment: "
            f"{deployment}"
        )

        client = (
            self.create_azure_client()
        )

        embedding = OpenAIEmbeddings(
            client=client,
            model=deployment,
        )

        return RAGASEmbeddingAdapter(
            embedding
        )

    # =========================================================
    # GENERATE DATASET
    # =========================================================

    def generate_dataset(self):

        print()
        print("=" * 70)
        print("GENERATING EVALUATION DATASET")
        print("=" * 70)

        samples = []

        for index, item in enumerate(
            EVALUATION_DATASET,
            start=1,
        ):

            question = item[
                "question"
            ]

            reference = item[
                "ground_truth"
            ]

            print()
            print("-" * 70)
            print(
                f"Question {index}"
            )
            print("-" * 70)

            print(question)

            # -------------------------------------------------
            # Run existing RAG pipeline
            # -------------------------------------------------

            response = (
                self.chatbot.ask(
                    question
                )
            )

            answer = response.get(
                "answer",
                "",
            )

            results = response.get(
                "results",
                [],
            )

            # -------------------------------------------------
            # Extract exact context used by LLM
            # -------------------------------------------------

            contexts = []

            for result in results:

                text = result.get(
                    "text",
                    "",
                )

                if text:

                    contexts.append(
                        text
                    )

            print(
                f"Answer length : "
                f"{len(answer)}"
            )

            print(
                f"Contexts      : "
                f"{len(contexts)}"
            )

            if not answer.strip():

                print(
                    "WARNING: Empty answer"
                )

            if not contexts:

                print(
                    "WARNING: No retrieved context"
                )

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference,
            )

            samples.append(
                sample
            )

        return EvaluationDataset(
            samples=samples
        )

    # =========================================================
    # EVALUATE
    # =========================================================

    def evaluate(self):

        dataset = (
            self.generate_dataset()
        )

        print()
        print("=" * 70)
        print("RUNNING RAGAS")
        print("=" * 70)

        evaluator_llm = (
            self.create_evaluator_llm()
        )

        evaluator_embeddings = (
            self.create_evaluator_embeddings()
        )

        metrics = [

            Faithfulness(
                llm=evaluator_llm
            ),

            ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                strictness=1,
            ),

            ContextPrecision(
                llm=evaluator_llm
            ),

            ContextRecall(
                llm=evaluator_llm
            ),
        ]

        print()
        print("Metrics:")

        for metric in metrics:

            print(
                f"  - {metric.name}"
            )

        print()
        print(
            "Starting RAGAS evaluation..."
        )

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            embeddings=(
                evaluator_embeddings
            ),
        )

        return result

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    def save_results(
        self,
        result,
    ):

        output_path = Path(
            OUTPUT_PATH
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if hasattr(
            result,
            "to_pandas",
        ):

            dataframe = (
                result.to_pandas()
            )

            records = (
                dataframe.to_dict(
                    orient="records"
                )
            )

        else:

            records = result

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                records,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print()
        print(
            "Results saved to:"
        )

        print(
            output_path
        )

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    def print_results(
        self,
        result,
    ):

        print()
        print("=" * 70)
        print("RAGAS RESULTS")
        print("=" * 70)

        if not hasattr(
            result,
            "to_pandas",
        ):

            print(result)

            return

        dataframe = (
            result.to_pandas()
        )

        print()

        print(
            dataframe.to_string(
                index=False
            )
        )

        # -----------------------------------------------------
        # Average scores
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print("AVERAGE SCORES")
        print("=" * 70)

        metric_names = [

            "faithfulness",

            "answer_relevancy",

            "context_precision",

            "context_recall",
        ]

        for metric in metric_names:

            if metric not in (
                dataframe.columns
            ):

                continue

            values = (
                dataframe[metric]
                .dropna()
            )

            if values.empty:

                print(
                    f"{metric:25s}: NaN"
                )

                continue

            print(
                f"{metric:25s}: "
                f"{values.mean():.4f}"
            )

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        result = (
            self.evaluate()
        )

        self.print_results(
            result
        )

        self.save_results(
            result
        )

        return result


# ============================================================
# MAIN
# ============================================================

def main():

    evaluator = (
        RAGASEvaluator()
    )

    evaluator.run()

    print()
    print("=" * 70)
    print(
        "RAGAS EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()