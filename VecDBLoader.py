import os
import time

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def process_document_in_vectorstore(vectorstore, document_list, embeddings, namespace):
    if not document_list:
        print("문서 리스트가 비어 있습니다.")
        return

    # 첫 번째 청크의 메타데이터를 사용하여 문서 수준에서 버전 비교
    first_chunk = document_list[0]
    source = first_chunk.metadata.get('source')
    version = first_chunk.metadata.get('version')

    # 벡터 스토어에서 해당 source를 가진 문서 검색
    # 메타데이터 기반 검색을 위해 metadata 필터 사용
    existing_docs = vectorstore.similarity_search(
        query=source,
        namespace=namespace,
        filter={'source': source},
        k=1
    )

    if existing_docs:
        existing_doc = existing_docs[0]
        existing_version = existing_doc.metadata.get('version')

        if existing_version == version:
            # 버전이 동일하므로 업데이트하지 않음
            print(f"문서 '{source}'는 이미 최신 버전입니다. 업데이트를 건너뜁니다.")
            return  # 함수 종료하여 추가 작업을 방지
        else:
            # 버전이 다르므로 기존 문서를 삭제하고 새로 추가
            print(f"문서 '{source}'의 버전이 변경되었습니다. 업데이트를 진행합니다.")
            # 벡터 스토어에서 해당 문서의 모든 청크 삭제
            vectorstore.delete(filter={'source': source}, namespace=namespace)
    else:
        print(f"문서 '{source}'는 새로운 문서입니다. 추가를 진행합니다.")

    # 분할된 청크들을 벡터 스토어에 추가
    vectorstore.add_documents(document_list, namespace=namespace)
    time.sleep(1)  # 필요한 경우 딜레이 추가


def embedding_text_line(source, text_list, document_version):
    # text_list가 리스트인지 확인
    if not isinstance(text_list, list):
        raise ValueError("text_list는 리스트여야 합니다.")

    # 리스트의 각 항목이 문자열인지 확인
    for text in text_list:
        if not isinstance(text, str):
            raise ValueError("리스트의 각 항목은 문자열이어야 합니다.")

    # 리스트를 하나의 문자열로 결합
    combined_text = " ".join(text_list)

    # Document 객체 생성 및 source 정보 추가
    document = Document(page_content=combined_text, metadata={"source": source, "version": document_version} if source else {})

    index_name = "wiki-upstage-index"
    chunk = 2000
    namespace = f"chunk_{chunk}_v2"

    # 텍스트 데이터를 분할하여 처리
    document_list = load_and_split_from_text([document], chunk)
    embeddings = get_embeddings()

    # 벡터 스토어 초기화
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        namespace=namespace,
        embedding=embeddings
    )

    # 문서를 벡터 스토어에 처리 (Metadata의 version 기반 처리)
    process_document_in_vectorstore(vectorstore, document_list, embeddings, namespace)


def embedding_file(file_path):
    loader = TextLoader(file_path, encoding='utf-8')

    index_name = "wiki-upstage-index"
    chunk = 1000
    namespace = f"chunk_{chunk}_v1"

    document_list = load_document_and_split(loader, chunk)
    embeddings = get_embeddings()

    PineconeVectorStore.from_documents(
        index_name=index_name,
        namespace=namespace,
        documents=document_list,
        embedding=embeddings
    )
    time.sleep(1)

def load_and_split_from_text(documents, chunk):
    """텍스트 데이터를 직접 처리하는 메서드"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    # Document 객체 리스트를 분할
    return splitter.split_documents(documents)

def load_document_and_split (loader, chunk):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return loader.load_and_split(text_splitter=splitter)

def source_reformat(document_list):
    if len(document_list) > 0:
        for doc in document_list:
            if 'source' in doc.metadata:
                file_name = os.path.basename(doc.metadata['source'])  # 파일명 추출
                doc.metadata['source'] = file_name  # 메타데이터 업데이트
    return document_list

def get_embeddings():
    return UpstageEmbeddings(model="solar-embedding-1-large")


# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import ServerlessSpec
# Pinecone 설정
# pc = Pinecone(
#     api_key=os.getenv("PINECONE_API_KEY")
# )
#
# index_name = "wiki-upstage-index"
# chunk = 500
# namespace = f"chunk_{chunk}_v1"
#
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=4096,
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )
#
# index = pc.Index(index_name)
#
# # 데이터 로드 및 분할
# loader = DirectoryLoader(".", glob="data/T*", show_progress=True)
#
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False
# )
#
# document_list = loader.load_and_split(text_splitter=text_splitter)
# print(f"총 {len(document_list)}개의 문서가 로드되었습니다.")
#
#
#
# # 모든 문서가 로드된 후 추가 작업 수행
# if len(document_list) > 0:
#     for doc in document_list:
#         if 'source' in doc.metadata:
#             file_name = os.path.basename(doc.metadata['source'])  # 파일명 추출
#             doc.metadata['source'] = file_name  # 메타데이터 업데이트
#
#     upstage_embedding = UpstageEmbeddings(model="solar-embedding-1-large")
#
#     database = PineconeVectorStore.from_documents(
#         index_name=index_name,
#         namespace=namespace,
#         documents=document_list,
#         embedding=upstage_embedding
#     )
#
# # 적절한 시간 대기
# time.sleep(1)


