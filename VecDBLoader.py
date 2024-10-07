import os
import time

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.utils.math import cosine_similarity
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


def get_vectorstore(embeddings, collection_name):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory='../chroma_db'  # 데이터베이스를 저장할 디렉토리 지정
    )

def embedding_text_line(source, text_list, document_version):
    # 텍스트 결합
    combined_text = " ".join(text_list)
    document = Document(
        page_content=combined_text,
        metadata={"source": source, "version": document_version}
    )

    collection_name = "wiki-upstage-collection"
    chunk_size = 1500

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings, collection_name)

    # 문서 분할
    document_list = load_and_split_from_text([document], chunk_size)

    # 메타데이터 기반으로 기존 문서 검색
    existing_docs = vectorstore.get(
        where={"source": source},
        include=['metadatas']
    )

    if existing_docs['metadatas']:
        existing_version = existing_docs['metadatas'][0].get('version')
        print(f"vector : {existing_version}")
        if existing_version == document_version:
            print(f"문서 '{source}'는 이미 최신 버전입니다. 업데이트를 건너뜁니다.")
            return
        else:
            print(f"문서 '{source}'의 버전이 변경되었습니다. 업데이트를 진행합니다.")
            # 기존 문서 삭제
            vectorstore.delete(where={"source": source})
    else:
        print(f"문서 '{source}'는 새로운 문서입니다. 추가를 진행합니다.")

    # 새로운 문서 추가
    vectorstore.add_documents(document_list)
    vectorstore.persist()
    time.sleep(1)


def embedding_text_line_pinecone(source, text_list):
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
    document = Document(page_content=combined_text, metadata={"source": source} if source else {})

    index_name = "wiki-upstage-index"
    chunk = 500
    namespace = f"chunk_{chunk}_v1"

    # 텍스트 데이터를 분할하여 처리
    document_list = load_and_split_from_text([document], chunk)
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
    return UpstageEmbeddings(model="solar-embedding-1-large-passage")


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


