import streamlit as st
# Import Settings directly from llama_index.core
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import config
from src.logger.logger import MyLogger
from src.db_handling import chroma_utils
import os # Import os for path manipulation

# Initialize logger (it's a singleton)
logger = MyLogger(config.LOG_NAME)
logger.setLevel(config.LOG_LEVEL) # Ensure level is set from config
logger.info("Streamlit App Session Starting") # Log start of a session


# --- Page Configuration (Set Title, Icon etc.) ---
st.set_page_config(
    page_title="Mars Tourism FAQ Bot",
    page_icon="🚀", # Can be an emoji or path to a favicon .ico file
    layout="wide"
)


@st.cache_resource(show_spinner="Connecting to Mars Knowledge Base...")
def load_knowledge_base():
    """Loads the existing VectorStoreIndex from ChromaDB."""

    ACTION = "Load knowledgebase (Streamlit)"
    logger.start(ACTION)

    idx = None

    try:
        logger.info("Attempting to load knowledge base for Streamlit app...")
        # 1. Configure LlamaIndex Settings (ensure API key is available)
        if not config.OPENAI_API_KEY:
             logger.critical("Streamlit: OPENAI_API_KEY not found in config!")
             st.error("OpenAI API Key not configured. Application cannot run.")
             st.stop()

        logger.debug("Configuring LlamaIndex settings for Streamlit session.")
        # Re-configuring settings here is acceptable for Streamlit's separate process
        Settings.llm = OpenAI(model=config.OPENAI_MODEL_NAME, api_key=config.OPENAI_API_KEY)
        Settings.embed_model = OpenAIEmbedding(model=config.EMBEDDING_MODEL_NAME, api_key=config.OPENAI_API_KEY)
        logger.debug("LlamaIndex settings configured.")

        # 2. Load the index using updated chroma_utils function
        logger.debug("Getting Chroma client and collection...")
        chroma_client = chroma_utils.get_chroma_client()
        chroma_collection = chroma_utils.get_or_create_chroma_collection(chroma_client) # Get existing collection
        logger.debug("Loading index structure from storage context...")
        # Use the correct function to get the index object
        idx, _ = chroma_utils.get_index_and_storage_context(chroma_collection)
        logger.info("Knowledge base index loaded successfully for Streamlit.")

    except Exception as e:
        st.error(f"Fatal error loading knowledge base: {e}. Please ensure the database is accessible and updated (run main.py).")
        logger.critical(f"Streamlit: Failed to load index: {e}", exc_info=True)
        st.stop() # Stop streamlit app if index cannot be loaded
    finally:
        logger.close(ACTION)

    return idx


# Load index once using caching
index = load_knowledge_base()

with st.sidebar:
    # Display Logo
    if os.path.exists(config.LOGO_PATH):
        st.image(config.LOGO_PATH, width=150)
    else:
        st.warning("Logo file not found at specified path in config.py")
        logger.warning(f"Logo file not found: {config.LOGO_PATH}")

    st.title("Mars Tourism Inc.")
    st.markdown("Your interstellar travel partner!")
    st.divider()
    # Add "Talk to Human" button here
    if st.button("❓ Talk to a Human Representative"):
        st.info("Thank you for your interest! Please contact Mars Tourism support at +1-MARS-EXPLORE for human assistance.")
        logger.info("User clicked 'Talk to Human' button.")
    st.divider()
    st.markdown("Powered by LlamaIndex & OpenAI")


# --- Main Chat Interface ---


st.header("🚀 Mars Tourism FAQ Bot 🚀")
st.caption("Ask me anything about your trip to Mars based on our FAQs!")


if index: # Only proceed if index loading was successful
    # Create the query engine (outside cache, uses cached index)
    try:
        query_engine = index.as_query_engine(
            streaming=False, # Set to True if you want streaming responses
            similarity_top_k=3
            # LLM with system prompt is set globally via Settings
        )
        logger.info("Query engine ready for user interaction.")
    except Exception as e:
        st.error(f"Failed to create query engine: {e}")
        logger.critical(f"Streamlit: Failed to create query engine: {e}", exc_info=True)
        st.stop()


    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized new chat session state.")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources associated with past assistant messages if stored
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                 with st.expander("Sources Cited", expanded=False):
                      st.markdown(message["sources"])


    # Accept user input using chat_input (appears at the bottom)
    if prompt := st.chat_input("Ask your question about Mars tourism..."):
        ACTION_QUERY = "PROCESS_USER_QUERY"
        logger.start(ACTION_QUERY)
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        logger.info(f"User query: {prompt}")

        # Display thinking message and query the engine
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            assistant_response_text = ""
            sources_markdown = "" # Initialize as empty string

            try:
                logger.debug("Querying the engine...")
                response = query_engine.query(prompt)
                assistant_response_text = response.response or "Sorry, I couldn't generate a response based on the available information."
                logger.info(f"Generated response (length: {len(assistant_response_text)})")

                # Display the actual response
                message_placeholder.markdown(assistant_response_text)

                # Prepare and potentially display source nodes
                if response.source_nodes:
                    logger.debug(f"Retrieved {len(response.source_nodes)} source nodes.")
                    sources_list = []
                    for i, node in enumerate(response.source_nodes):
                        # Use updated metadata keys
                        file_path = node.metadata.get('file_path', 'Unknown Path')
                        # Extract filename from path for display
                        file_name = os.path.basename(file_path)
                        subject = node.metadata.get('subject', 'N/A')
                        question = node.metadata.get('question', 'N/A') # Question is still available
                        score = node.score or 0.0
                        # Store source info including subject
                        sources_list.append(
                            f"**{i+1}. `{file_name}`** (Score: {score:.3f})\n"
                            f"   *Subject:* {subject}\n"
                            f"   *Q:* {question}\n"
                        )
                    sources_markdown = "---\n" + "\n".join(sources_list)

                    # Display sources in an expander below the response
                    with st.expander("Sources Used", expanded=False):
                         st.markdown(sources_markdown)
                else:
                    logger.info("No source nodes retrieved for this query.")
                    # sources_markdown remains empty or you could set it explicitly:
                    # sources_markdown = "No specific sources cited."

            except Exception as e:
                assistant_response_text = f"Sorry, an error occurred: {e}"
                message_placeholder.error(assistant_response_text)
                logger.error(f"Error during query engine processing: {e}", exc_info=True)
                sources_markdown = "*Error occurred during source retrieval.*" # Indicate error in stored sources


            # Add assistant response AND sources to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response_text,
                "sources": sources_markdown # Store sources markdown (can be empty)
            })

        logger.close(ACTION_QUERY) # Close query action

else:
    st.error("Knowledge base index could not be loaded. Application cannot function.")
    logger.critical("Streamlit app cannot function because index is None.")

logger.info("Streamlit App Session Render Complete")
