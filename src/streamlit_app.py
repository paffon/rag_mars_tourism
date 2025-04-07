import streamlit as st
import os
from typing import Optional
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.base import BaseIndex
from llama_index.query_engine import BaseQueryEngine
from llama_index.response.schema import Response

import config
from src.logger.logger import MyLogger
from src.db_handling import chroma_utils


logger = MyLogger(config.LOG_NAME)
logger.setLevel(config.LOG_LEVEL)
logger.info("Streamlit App Session Starting")

st.set_page_config(
    page_title="Mars Tourism FAQ Bot",
    page_icon="🚀",
    layout="wide"
)


def validate_api_key() -> None:
    if not config.OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY not found in config!")
        st.error("OpenAI API Key not configured. Application cannot run.")
        st.stop()


def configure_llama_settings() -> None:
    Settings.llm = OpenAI(model=config.OPENAI_MODEL_NAME, api_key=config.OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model=config.EMBEDDING_MODEL_NAME, api_key=config.OPENAI_API_KEY)


@st.cache_resource(show_spinner="Connecting to Mars Knowledge Base...")
def load_knowledge_base() -> Optional[BaseIndex]:
    ACTION = "Load knowledgebase"
    logger.start(ACTION)

    try:
        validate_api_key()
        configure_llama_settings()

        chroma_client = chroma_utils.get_chroma_client()
        chroma_collection = chroma_utils.get_or_create_chroma_collection(chroma_client)
        index, _ = chroma_utils.get_index_and_storage_context(chroma_collection)
        logger.info("Knowledge base loaded.")
        return index

    except Exception as e:
        logger.critical(f"Failed to load index: {e}", exc_info=True)
        st.error(f"Fatal error loading knowledge base: {e}. Run main.py to update the database.")
        st.stop()

    finally:
        logger.close(ACTION)


def display_sidebar() -> None:
    with st.sidebar:
        if os.path.exists(config.LOGO_PATH):
            st.image(config.LOGO_PATH, width=150)
        else:
            st.warning("Logo not found.")
            logger.warning(f"Logo missing at: {config.LOGO_PATH}")

        st.title("Mars Tourism Inc.")
        st.markdown("Your interstellar travel partner!")
        st.divider()

        if st.button("❓ Talk to a Human Representative"):
            st.info("Contact Mars Tourism support at +1-MARS-EXPLORE.")
            logger.info("User clicked 'Talk to Human'.")

        st.divider()
        st.markdown("Powered by LlamaIndex & OpenAI")


def render_sources(response: Response) -> str:
    sources = []
    for i, node in enumerate(response.source_nodes):
        file_name = os.path.basename(node.metadata.get("file_path", "Unknown"))
        subject = node.metadata.get("subject", "N/A")
        question = node.metadata.get("question", "N/A")
        score = node.score or 0.0

        sources.append(
            f"**{i+1}. `{file_name}`** (Score: {score:.3f})\n"
            f"   *Subject:* {subject}\n"
            f"   *Q:* {question}\n"
        )

    return "---\n" + "\n".join(sources) if sources else ""


def process_user_prompt(prompt: str, query_engine: BaseQueryEngine) -> tuple[str, str]:
    ACTION = "PROCESS_USER_QUERY"
    logger.start(ACTION)

    try:
        response = query_engine.query(prompt)
        answer = response.response or "No response generated."
        logger.info("Query processed successfully.")
        sources_markdown = render_sources(response)
        return answer, sources_markdown

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return f"Error: {e}", "*An error occurred retrieving sources.*"

    finally:
        logger.close(ACTION)


def main():
    index = load_knowledge_base()
    if not index:
        return

    display_sidebar()
    st.header("🚀 Mars Tourism FAQ Bot 🚀")
    st.caption("Ask me anything about your trip to Mars!")

    try:
        query_engine = index.as_query_engine(streaming=False, similarity_top_k=3)
    except Exception as e:
        logger.critical(f"Failed to create query engine: {e}", exc_info=True)
        st.error(f"Error: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources Cited", expanded=False):
                    st.markdown(msg["sources"])

    if prompt := st.chat_input("Ask your question about Mars tourism..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            response_text, sources_md = process_user_prompt(prompt, query_engine)
            placeholder.markdown(response_text)

            if sources_md:
                with st.expander("Sources Used", expanded=False):
                    st.markdown(sources_md)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources_md
        })

    logger.info("Streamlit App Session Render Complete")


if __name__ == "__main__":
    main()
