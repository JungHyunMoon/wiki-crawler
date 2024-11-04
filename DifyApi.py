import os

import requests
from typing import Dict, Any

from dotenv import load_dotenv

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

def get_documents()-> Dict[str, Any]:
    # API 엔드포인트 설정
    url = f"{api_url}/v1/datasets/{dataset_id}/documents"

    print(url)

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

doc_title_list = {}
documents = get_documents().get("data")
for document in documents:
    # 딕셔너리로 저장하여 key:value 형식 유지
    doc_title_list[document["name"]] = document["id"]


# test code
param = "회고록2"
last_modified = "지난주 화요일 오후 5:45"
text_data = "수정한다요"
if param in doc_title_list.keys() and last_modified.startswith("지난주"):
    update_by_text(param, text_data, doc_title_list[param])

if param not in doc_title_list.keys():
    create_by_text(param, text_data)
