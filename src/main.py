import subprocess
import time
import webbrowser
import sys
import os
import socket

# Ensure the 'src' directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.insert(0, project_root)

import config
from src.logger.logger import MyLogger
# Ensure the correct module name is imported
from src.db_handling import indexing

# Initialize logger using settings from config
logger = MyLogger(config.LOG_NAME)
logger.setLevel(config.LOG_LEVEL)


def is_streamlit_running(url: str) -> bool:
    """
    Checks if the Streamlit server is running at the given URL.
    """
    ACTION = "CHECK_STREAMLIT"
    logger.start(ACTION)
    try:
        # Parse the host and port from the URL
        host_port = url.split("//")[1].split(":")
        host = host_port[0]
        port = int(host_port[1])

        # Create a socket
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        logger.info(f"Streamlit is reachable at {url}")
        return True
    except socket.error as e:
        logger.warning(f"Streamlit not yet reachable at {url}: {e}")
        return False
    finally:
        logger.close(ACTION)


def run_streamlit_app():
    """Starts the Streamlit application and waits for it to be ready."""

    ACTION = "STREAMLIT"
    logger.start(ACTION)
    streamlit_app_path = os.path.join(current_dir, "streamlit_app.py")

    streamlit_command = [
        sys.executable,
        "-m", "streamlit", "run", streamlit_app_path,
        "--server.port", str(config.STREAMLIT_SERVER_PORT),
    ]

    process = subprocess.Popen(streamlit_command)
    logger.info(f"Streamlit process started (PID: {process.pid}). Waiting for server to start...")

    max_attempts = 20  # Maximum number of times to check
    wait_interval = 3  # Wait 3 seconds between checks
    for attempt in range(1, max_attempts + 1):
        if is_streamlit_running(config.STREAMLIT_APP_URL):
            logger.info("Streamlit server is ready.")
            break
        logger.info(f"Attempt {attempt}/{max_attempts}: Checking Streamlit...")
        time.sleep(wait_interval)
    else:
        logger.error("Streamlit server did not start in a reasonable time. Skipping auto-open.")
        logger.close(ACTION) # Close here since we're exiting
        return

    logger.info(f"Opening Streamlit app in browser at {config.STREAMLIT_APP_URL}")
    try:
        webbrowser.open(config.STREAMLIT_APP_URL)
    except Exception as browser_e:
        logger.warning(f"Could not automatically open browser: {browser_e}. Please navigate to {config.STREAMLIT_APP_URL} manually.")

    logger.info("Streamlit launch initiated. main.py will now exit.")
    logger.close(ACTION)


if __name__ == "__main__":
    APP_ACTION = "MAIN Application Startup"
    logger.start(APP_ACTION)

    try:
        indexing.synchronize_vector_db()
        logger.info("Database synchronization complete. Starting Streamlit app...")
        run_streamlit_app()

    except Exception as critical_err:
        logger.critical(f"A critical error occurred: {critical_err}", exc_info=True)

    finally:
        logger.close(APP_ACTION)
