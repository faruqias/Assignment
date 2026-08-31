import os
import time

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()


class AzureOpenAIClient:
    """
    Adapter around Azure OpenAI Chat Completions.

    Designed for GPT-5.x deployments.

    Supports:
        - Complete response generation
        - Streaming response generation
    """

    def __init__(self):

        endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )

        api_key = os.getenv(
            "AZURE_OPENAI_API_KEY"
        )

        api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION"
        )

        self.deployment = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT"
        )

        # -----------------------------------------------------
        # Validate configuration
        # -----------------------------------------------------

        if not endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is not configured."
            )

        if not api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is not configured."
            )

        if not api_version:
            raise ValueError(
                "AZURE_OPENAI_API_VERSION is not configured."
            )

        if not self.deployment:
            raise ValueError(
                "AZURE_OPENAI_CHAT_DEPLOYMENT is not configured."
            )

        # -----------------------------------------------------
        # Client
        # -----------------------------------------------------

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        print(
            f"Azure OpenAI Deployment: "
            f"{self.deployment}"
        )

    # =========================================================
    # GENERATE
    # =========================================================

    def generate(
        self,
        prompt,
        model=None,
        temperature=0.1,
        max_tokens=200,
    ):
        """
        Generate a complete response.

        GPT-5.x:
            max_tokens -> max_completion_tokens
            temperature is not sent.
        """

        start_time = time.perf_counter()

        deployment = (
            model
            or self.deployment
        )

        print()
        print(
            "Sending prompt to Azure OpenAI..."
        )

        print(
            f"Model      : {deployment}"
        )

        print(
            f"Prompt     : "
            f"{len(prompt):,} characters"
        )

        response = (
            self.client
            .chat
            .completions
            .create(
                model=deployment,

                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],

                max_completion_tokens=max_tokens,
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Azure OpenAI total: "
            f"{elapsed:.3f} sec"
        )

        return (
            response
            .choices[0]
            .message
            .content
            or ""
        )

    # =========================================================
    # STREAM
    # =========================================================

    def stream(
        self,
        prompt,
        model=None,
        temperature=0.1,
        max_tokens=200,
    ):
        """
        Stream response tokens from Azure OpenAI.

        GPT-5.x:
            max_tokens -> max_completion_tokens
            temperature is not sent.
        """

        start_time = time.perf_counter()
        first_token_time = None

        deployment = (
            model
            or self.deployment
        )

        print()
        print(
            "Streaming from Azure OpenAI..."
        )

        print(
            f"Model      : {deployment}"
        )

        print(
            f"Prompt     : "
            f"{len(prompt):,} characters"
        )

        response_stream = (
            self.client
            .chat
            .completions
            .create(
                model=deployment,

                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],

                max_completion_tokens=max_tokens,

                stream=True,
            )
        )

        for response in response_stream:

            if not response.choices:
                continue

            delta = (
                response
                .choices[0]
                .delta
            )

            token = (
                delta.content
                or ""
            )

            if not token:
                continue

            if first_token_time is None:

                first_token_time = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"First token: "
                    f"{first_token_time:.3f} sec"
                )

            yield token

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Azure OpenAI total: "
            f"{elapsed:.3f} sec"
        )