# Mars Tourism FAQ Bot
![image](https://github.com/user-attachments/assets/0e1cd18a-4506-450a-92f9-f3d8c1df0fe7)

This project is a Streamlit application that provides a chatbot to answer questions about Mars tourism, using information from a collection of FAQ documents. It leverages LlamaIndex and OpenAI to provide context-aware responses.

## Features

* **Chat Interface:** A user-friendly interface to ask questions about Mars trips.
* **Contextual Answers:** Answers are based on provided FAQ documents.
* **Source Tracking:** The bot cites the source documents used to generate responses.
* **Logging:** Detailed logging of application behavior.

## Installation

1.  **Python Version:** Python 3.8 or later is recommended.

2.  **Clone the repository:**

    ```bash
    git clone <your_repository_url>
    cd rag_mars_tourism
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**

    * You **must** set the `OPENAI_API_KEY` environment variable.
    * It is recommended to use a `.env` file. Create a file named `.env` in the project root with the following content:

        ```
        OPENAI_API_KEY=YOUR_OPENAI_API_KEY
        ```

    * If you use a `.env` file, you might need to install `python-dotenv`:

        ```bash
        pip install python-dotenv
        ```

5.  **Data:**

    * FAQ documents should be placed in the `data/` directory.
    * The application expects `.txt` files with a specific format (see `src/db_handling/parsing.py`).

## Usage

1.  **Run the main script:**

    ```bash
    python main.py
    ```

    * This script will:
        * Synchronize the FAQ documents with the ChromaDB database.
        * Start the Streamlit application.
        * The Streamlit app should open automatically in your browser. If not, you can access it at `http://localhost:8501`.

## Configuration

* `config.py`:  Contains application settings such as:
    * `OPENAI_MODEL_NAME`: The OpenAI model used for generating responses.
    * `EMBEDDING_MODEL_NAME`: The OpenAI model used for generating embeddings.
    * `CHROMA_COLLECTION_NAME`: The name of the ChromaDB collection.
    * `DATA_DIR`:  The directory where FAQ documents are stored.
    * `PERSIST_DIR`:  The directory where the ChromaDB database is stored.
    * `LOG_LEVEL`:  The logging level.

## Data Format

* FAQ documents are expected to be plain text files (`.txt`).
* Each file should be formatted as follows:

    ```text
    Subject/Topic
    Question 1?
    Answer 1
    Question 2?
    Answer 2
    ...
    ```

* The first line is the subject.
* Questions must end with a question mark.
* Empty lines are ignored.

## License

Feel free to use this code for anything you like.

## Author

(Omri Nardi Niri / paffon)
