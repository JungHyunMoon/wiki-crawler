# import os
# import time
#
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_pinecone import PineconeVectorStore
# from pinecone.grpc import PineconeGRPC as Pinecone
# from langchain_upstage import UpstageEmbeddings
# from dotenv import load_dotenv
# from Logger import create_wiki_log
#
# load_dotenv()
#
#
# def embedding_text_line(source, text_list, document_version):
#     # 텍스트 결합
#     combined_text = " ".join(text_list)
#     document = Document(
#         page_content=combined_text,
#         metadata={"source": source, "version": document_version}
#     )
#     chunk_size = 700
#     # 문서 분할
#     document_list = load_and_split_from_text(source, [document], chunk_size)
#
#     index_name = "wiki-upstage-index"
#     collection_name = f"chunk_{chunk_size}_v1"
#
#     # DB 초기화
#     pc = Pinecone()
#     index = pc.Index(index_name)
#
#     filter_condition = {
#         "source": source,
#         "version": document_version
#     }
#
#     # 인덱스의 차원에 맞춘 제로 벡터 사용
#
#     vector_dimension = 4096
#     dummy_vector = [0.0] * vector_dimension
#
#     # 기존 문서 조회 쿼리
#     existing_docs = index.query(
#         namespace=collection_name,
#         vector=dummy_vector,
#         filter=filter_condition,
#         include_metadata=True,  # 메타데이터 포함
#         top_k=50
#     )
#
#     log_msg = ""
#
#     # metadata version을 기준으로 문서 갱신 판단
#     if existing_docs.get("matches"):
#         existing_version = existing_docs.get("matches")[0].get("metadata").get("version")
#         if existing_version == document_version:
#             log_msg = f"EXIST >>> 문서 '{source}'는 이미 최신 버전입니다. 업데이트를 건너뜁니다."
#             create_wiki_log("existFile", log_msg)
#             return
#         else:
#             log_msg = f"UPDATE >>> 문서 '{source}'의 버전이 변경되었습니다. 기존 문서를 삭제하고 업데이트를 진행합니다."
#             # ID 리스트 추출
#             ids_to_delete = [match['id'] for match in existing_docs["matches"]]
#             # 기존 문서 삭제
#             # ID 리스트가 비어있지 않은 경우에만 삭제
#             if ids_to_delete:
#                 index.delete(ids=ids_to_delete, namespace=collection_name)
#     else:
#         log_msg = f"CREATE >>> 문서 '{source}'는 새로운 문서입니다. 추가를 진행합니다."
#
#     # 로그 남기기
#     create_wiki_log("newFile", log_msg)
#
#     # 새로운 문서 추가
#     PineconeVectorStore.from_documents(
#         index_name=index_name,
#         namespace=collection_name,
#         documents=document_list,
#         embedding=get_embeddings()
#     )
#     time.sleep(1)
#
#
# def embedding_text_line_pinecone(source, text_list):
#     # text_list가 리스트인지 확인
#     if not isinstance(text_list, list):
#         raise ValueError("text_list는 리스트여야 합니다.")
#
#     # 리스트의 각 항목이 문자열인지 확인
#     for text in text_list:
#         if not isinstance(text, str):
#             raise ValueError("리스트의 각 항목은 문자열이어야 합니다.")
#
#     # 리스트를 하나의 문자열로 결합
#     combined_text = " ".join(text_list)
#
#     # Document 객체 생성 및 source 정보 추가
#     document = Document(page_content=combined_text, metadata={"source": source} if source else {})
#
#     index_name = "wiki-upstage-index"
#     chunk = 500
#     namespace = f"chunk_{chunk}_v1"
#
#     # 텍스트 데이터를 분할하여 처리
#     document_list = load_and_split_from_text(source, [document], chunk)
#     embeddings = get_embeddings()
#
#     PineconeVectorStore.from_documents(
#         index_name=index_name,
#         namespace=namespace,
#         documents=document_list,
#         embedding=embeddings
#     )
#     time.sleep(1)
#
#
#
# def load_and_split_from_text(source, documents, chunk):
#     """텍스트 데이터를 직접 처리하는 메서드"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk,
#         chunk_overlap=200,
#         length_function=len,
#         is_separator_regex=False
#     )
#     # Document 객체 리스트를 분할
#     document_list = splitter.split_documents(documents)
#
#     # 각 청크의 첫 줄에 source 추가
#     # Hallucination 방지 강화
#     for doc in document_list:
#         doc.page_content = f"{source}\n\n" + doc.page_content
#
#     return document_list
#
# def load_document_and_split (loader, chunk):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk,
#         chunk_overlap=chunk / 5,
#         length_function=len,
#         is_separator_regex=False
#     )
#     return loader.load_and_split(text_splitter=splitter)
#
# def source_reformat(document_list):
#     if len(document_list) > 0:
#         for doc in document_list:
#             if 'source' in doc.metadata:
#                 file_name = os.path.basename(doc.metadata['source'])  # 파일명 추출
#                 doc.metadata['source'] = file_name  # 메타데이터 업데이트
#     return document_list
#
# def get_embeddings():
#     return UpstageEmbeddings(model="solar-embedding-1-large-passage")
