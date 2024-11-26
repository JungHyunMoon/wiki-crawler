import os
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from Logger import create_wiki_log

# 환경 변수 로드 및 검증
load_dotenv()
API_URL = os.getenv("API_URL")
DATASET_ID = os.getenv("DATASET_ID")
API_KEY = os.getenv("API_KEY")

if not all([API_URL, DATASET_ID, API_KEY]):
    raise EnvironmentError("API_URL, DATASET_ID, 또는 API_KEY가 설정되지 않았습니다.")

# 공통 요청 헤더
COMMON_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def send_request(method: str, endpoint: str, data: Optional[dict] = None) -> Dict[str, Any]:
    """
    재사용 가능한 API 요청 헬퍼 함수.
    """
    url = f"{API_URL}{endpoint}"
    try:
        response = requests.request(method, url, headers=COMMON_HEADERS, json=data)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
        return response.json()
    except requests.RequestException as e:
        log_msg = f"ERROR >>> API 호출 실패: {e}"
        create_wiki_log("error", log_msg)
        return {
            "status": "error",
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None)
        }

def create_by_text(main_title: str, text_data: str) -> None:
    """
    새 문서를 생성하는 함수.
    """
    data = {
        "name": main_title,
        "text": text_data,
        "indexing_technique": "high_quality",
        "process_rule": {"mode": "automatic"}
    }
    response = send_request("POST", f"/v1/datasets/{DATASET_ID}/document/create_by_text", data)
    if response.get("status") == "error":
        log_msg = f"ERROR >>> 문서 생성 실패: '{main_title}'"
        create_wiki_log("error", log_msg)
    else:
        log_msg = f"CREATE >>> 문서 '{main_title}'은(는) 새로운 문서입니다. 추가를 진행합니다."
        create_wiki_log("success", log_msg)

def get_datasets() -> Dict[str, Any]:
    """
    데이터셋 목록을 가져오는 함수.
    """
    return send_request("GET", "/v1/datasets")

def get_documents(page: int) -> Dict[str, Any]:
    """
    데이터셋 내 문서 목록을 페이지 단위로 가져오는 함수.
    """
    return send_request("GET", f"/v1/datasets/{DATASET_ID}/documents?page={page}&limit=100")

def update_by_text(main_title: str, text_data: str, document_id: str) -> None:
    """
    기존 문서를 업데이트하는 함수.
    """
    data = {
        "name": main_title,
        "text": text_data,
        "indexing_technique": "high_quality",
        "process_rule": {"mode": "automatic"}
    }
    response = send_request("POST", f"/v1/datasets/{DATASET_ID}/documents/{document_id}/update_by_text", data)
    if response.get("status") == "error":
        log_msg = f"ERROR >>> 문서 업데이트 실패: '{main_title}'"
        create_wiki_log("error", log_msg)
    else:
        log_msg = f"UPDATE >>> 문서 '{main_title}'이(가) 성공적으로 업데이트되었습니다."
        create_wiki_log("success", log_msg)

def save_doc(main_title: str, text_data: str, last_modified: str, exist_doc: Dict[str, str]) -> None:
    """
    문서가 존재하면 업데이트, 없으면 생성하는 함수.
    """
    log_msg: str
    if main_title in exist_doc and last_modified.startswith("지난주"):
        update_by_text(main_title, text_data, exist_doc[main_title])
        log_msg = f"UPDATE >>> 문서 '{main_title}'의 버전이 {last_modified} 일자로 변경되었습니다."
    elif main_title not in exist_doc:
        create_by_text(main_title, text_data)
        log_msg = f"CREATE >>> 문서 '{main_title}'은(는) 새로운 문서입니다. 추가를 진행합니다."
    else:
        log_msg = f"EXIST >>> 문서 '{main_title}'은(는) 이미 최신 버전입니다. 업데이트를 건너뜁니다."

    create_wiki_log("crawl", log_msg)
