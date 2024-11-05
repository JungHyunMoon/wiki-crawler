import math
import os

import requests
from typing import Dict, Any

from dotenv import load_dotenv
from Logger import create_wiki_log

load_dotenv()
api_url = os.environ.get("API_URL")
dataset_id = os.environ.get("DATASET_ID")
api_key = os.environ.get("API_KEY")

def create_by_text(main_title, text_data):
    # API 엔드포인트 설정
    url = f"{api_url}/v1/datasets/{dataset_id}/document/create_by_text"

    # 요청 헤더 설정
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 요청 데이터 설정
    data = {
        "name": main_title,  # main_title이 name 필드에 들어갑니다
        "text": text_data,  # text_data가 text 필드에 들어갑니다
        "indexing_technique": "high_quality",
        "process_rule": {
            "mode": "automatic"
        }
    }

    # API 호출
    # 응답 처리 할지 보류
    response = requests.post(url, headers=headers, json=data)

def get_datasets()-> Dict[str, Any]:
    url = f"{api_url}/v1/datasets"

    # 요청 헤더 설정
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    # 응답 처리
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "status": "error",
            "status_code": response.status_code,
            "error": response.text
        }

def get_documents(page)-> Dict[str, Any]:
    # API 엔드포인트 설정
    url = f"{api_url}/v1/datasets/{dataset_id}/documents?page={page}&limit=100"

    # 요청 헤더 설정
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # API 호출
    response = requests.get(url, headers=headers)

    # 응답 처리
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "status": "error",
            "status_code": response.status_code,
            "error": response.text
        }


def update_by_text(main_title, text_data, document_id):
    # API 엔드포인트 설정
    url = f"{api_url}/v1/datasets/{dataset_id}/documents/{document_id}/update_by_text"

    # 요청 헤더 설정
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 요청 데이터 설정
    data = {
        "name": main_title,  # main_title이 name 필드에 들어갑니다
        "text": text_data,  # text_data가 text 필드에 들어갑니다
        "indexing_technique": "high_quality",
        "process_rule": {
            "mode": "automatic"
        }
    }

    # API 호출
    # 응답 처리 할지 보류
    response = requests.post(url, headers=headers, json=data)


def save_doc(main_title, text_data, last_modified, exist_doc):
    log_msg = f"EXIST >>> 문서 '{main_title}'은(는) 이미 최신 버전입니다. 업데이트를 건너뜁니다."

    if main_title in exist_doc.keys() and last_modified.startswith("지난주"):
        update_by_text(main_title, text_data, exist_doc[main_title])
        log_msg = f"UPDATE >>> 문서 '{main_title}'의 버전이 {last_modified} 일자로 변경되었습니다. 최신화를 진행합니다."

    if main_title not in exist_doc.keys():
        create_by_text(main_title, text_data)
        log_msg = f"CREATE >>> 문서 '{main_title}'은(는) 새로운 문서입니다. 추가를 진행합니다."
        

    create_wiki_log("crawl", log_msg)


