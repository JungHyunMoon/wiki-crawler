import logging
import os
from datetime import datetime
from pytz import timezone

def create_wiki_log(process_name, message):
    # 한국 시간대 설정
    seoul_tz = timezone("Asia/Seoul")
    current_date = datetime.now(seoul_tz).strftime("%Y-%m-%d")
    log_file_name = f"{process_name}_{current_date}.log"
    log_file_path = os.path.join("logs", log_file_name)

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 로거 설정
    logger = logging.getLogger(f"{process_name}Logger")
    logger.setLevel(logging.INFO)

    # 핸들러가 이미 추가되어 있는지 확인
    if not logger.handlers:
        handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # 한국 시간으로 메시지 기록
    logger.info(message)
