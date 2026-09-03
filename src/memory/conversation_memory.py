from collections import deque


class ConversationMemory:
    """
    Maintains the last N conversation interactions.

    One interaction consists of:

        user question
        assistant answer

    Default:
        last 4 interactions
    """

    def __init__(
        self,
        max_interactions=4
    ):

        self.max_interactions = (
            max_interactions
        )

        self.history = deque(
            maxlen=max_interactions
        )

    # =========================================================
    # ADD INTERACTION
    # =========================================================

    def add(
        self,
        question,
        answer
    ):

        self.history.append(
            {
                "question": question,
                "answer": answer
            }
        )

    # =========================================================
    # GET HISTORY
    # =========================================================

    def get_history(self):

        return list(
            self.history
        )

    # =========================================================
    # FORMAT HISTORY
    # =========================================================

    def format_history(self):

        if not self.history:

            return (
                "No previous conversation."
            )

        parts = []

        for index, interaction in enumerate(
            self.history,
            start=1
        ):

            parts.append(
                f"Interaction {index}:\n"
                f"User: "
                f"{interaction['question']}\n"
                f"Assistant: "
                f"{interaction['answer']}"
            )

        return "\n\n".join(
            parts
        )

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self):

        self.history.clear()

    # =========================================================
    # COUNT
    # =========================================================

    def count(self):

        return len(
            self.history
        )