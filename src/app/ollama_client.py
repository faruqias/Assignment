import ollama


class OllamaClient:
    """
    Small adapter around the Ollama Python client.
    """

    def generate(
        self,
        prompt,
        model,
        temperature,
        max_tokens,
    ):
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        return response["message"]["content"]

    def stream(
        self,
        prompt,
        model,
        temperature,
        max_tokens,
    ):
        response_stream = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        for response in response_stream:

            token = (
                response
                .get("message", {})
                .get("content", "")
            )

            if token:
                yield token