import gradio as gr

from src.app.application import RAGApplication


# =========================================================
# APPLICATION
# =========================================================

app = RAGApplication()

def clear_conversation():
    app.chatbot.memory.clear()
    return []


# =========================================================
# CSS
# =========================================================

css = """
body {
    background: #f5f5f5;
}

#upload-panel {
    background: white;
    border-radius: 12px;
    padding: 20px;
}

#chat-panel {
    background: white;
    border-radius: 12px;
    padding: 20px;
}

#status-box {
    background: #f5f5f5;
}

#chatbot {
    background: white;
}
"""


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="Enterprise RAG Assistant"
) as demo:

    # =====================================================
    # HEADER
    # =====================================================

    gr.Markdown(
        """
        # Enterprise RAG Assistant

        Upload your PDF documents and ask questions
        using document-grounded AI.
        """
    )

    # =====================================================
    # MAIN
    # =====================================================

    with gr.Row():

        # =================================================
        # UPLOAD
        # =================================================

        with gr.Column(
            scale=1,
            elem_id="upload-panel"
        ):

            gr.Markdown(
                "### 📄 Documents"
            )

            files = gr.File(
                label="Upload PDF Documents",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath"
            )

            upload_button = gr.Button(
                "Upload Documents",
                variant="primary"
            )

            status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=8,
                elem_id="status-box"
            )

        # =================================================
        # CHAT
        # =================================================

        with gr.Column(
            scale=2,
            elem_id="chat-panel"
        ):

            gr.Markdown(
                "### 💬 Ask Questions"
            )

            chatbot = gr.Chatbot(
                label="RAG Assistant",
                height=500,
                render_markdown=True
            )

            # clear_chat = gr.Button(
            #     "Clear Conversation"
            # )

            # clear_chat.click(
            #     fn=clear_conversation,
            #     inputs=None,
            #     outputs=chatbot
            # )

            message = gr.Textbox(
                label="Question",
                placeholder=(
                    "Ask a question about your documents..."
                ),
                lines=2
            )

            with gr.Row():

                ask_button = gr.Button(
                    "Ask",
                    variant="primary"
                )

                clear_button = gr.Button(
                    "Clear"
                )

    # =========================================================
    # UPLOAD EVENT
    # =========================================================

    upload_button.click(
        fn=app.upload_documents,
        inputs=files,
        outputs=status
    )

    # =========================================================
    # ASK BUTTON
    # =========================================================

    ask_button.click(
        fn=app.ask_question,
        inputs=[
            message,
            chatbot
        ],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=message
    )

    # =========================================================
    # ENTER KEY
    # =========================================================

    message.submit(
        fn=app.ask_question,
        inputs=[
            message,
            chatbot
        ],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=message
    )

    # =========================================================
    # CLEAR
    # =========================================================

    clear_button.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot
    )


# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":

    demo.queue()

    demo.launch(
        css=css,
        theme=gr.themes.Default(
            primary_hue="orange"
        )
    )