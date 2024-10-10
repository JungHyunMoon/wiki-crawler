import logging
import os
from datetime import datetime

def create_wiki_log(process_name, message):
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_name= f"{process_name}_{current_date}.log"
    log_file_path = os.path.join("logs", log_file_name)

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger(f"{process_name}Logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.info(message)

