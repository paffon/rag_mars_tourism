import subprocess
import time
import webbrowser
import sys
import os
import socket

# Ensure the 'src' directory is in the Python path
current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_root: str = current_dir
sys.path.insert(0, project_root)

import config
from src.logger.logger import MyLogger
from src.db_handling import indexing

# Initialize logger
logger = MyLogger(config.LOG_NAME)
logger.setLevel(config.LOG_LEVEL)


def is_streamlit_running(url: str) -> bool:
    """
    Check if the Streamlit server is reachable at a given URL.

    :param url: Full URL where the Streamlit server is expected to run (e.g., "http://localhost:8501")
    :return: True if the server responds to a socket connection; False otherwise
    """
    ACTION = "CHECK_STREAMLIT"
    logger.start(ACTION)
    try:
        host, port = url.split("//")[1].split(":")
        sock = socket.create_connection((host, int(port)), timeout=5)
        sock.close()
        logger.info(f"Streamlit is reachable at {url}")
        return True
    except socket.error as e:
        logger.warning(f"Streamlit not yet reachable at {url}: {e}")
        return False
    finally:
        logger.close(ACTION)


def run_streamlit_app() -> None:
    """
    Launch the Streamlit app and open it in the default web browser if reachable.
    Logs attempts and waits until the app is running or timeout occurs.
    """
    ACTION = "STREAMLIT"
    logger.start(ACTION)

    streamlit_app_path: str = os.path.join(current_dir, "streamlit_app.py")
    streamlit_command: list[str] = [
        sys.executable,
        "-m", "streamlit", "run", streamlit_app_path,
        "--server.port", str(config.STREAMLIT_SERVER_PORT),
    ]

    process = subprocess.Popen(streamlit_command)
    logger.info(f"Streamlit process started (PID: {process.pid}). Waiting for server to start...")

    max_attempts = 20
    wait_interval = 3  # seconds

    for attempt in range(1, max_attempts + 1):
        if is_streamlit_running(config.STREAMLIT_APP_URL):
            logger.info("Streamlit server is ready.")
            break
        logger.info(f"Attempt {attempt}/{max_attempts}: Checking Streamlit...")
        time.sleep(wait_interval)
    else:
        logger.error("Streamlit server did not start in a reasonable time. Skipping auto-open.")
        logger.close(ACTION)
        return

    logger.info(f"Opening Streamlit app in browser at {config.STREAMLIT_APP_URL}")
    try:
        webbrowser.open(config.STREAMLIT_APP_URL)
    except Exception as e:
        logger.warning(f"Could not automatically open browser: {e}. Please navigate to {config.STREAMLIT_APP_URL} manually.")

    logger.info("Streamlit launch initiated. main.py will now exit.")
    logger.close(ACTION)


if __name__ == "__main__":
    APP_ACTION = "MAIN Application Startup"
    logger.start(APP_ACTION)

    try:
        indexing.synchronize_vector_db()
        logger.info("Database synchronization complete. Starting Streamlit app...")
        run_streamlit_app()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        logger.close(APP_ACTION)
