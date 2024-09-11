import os
import time

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv

load_dotenv()

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


