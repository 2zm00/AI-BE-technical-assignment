import logging
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from app.core.config import settings

logger = logging.getLogger(__name__)

# LangChain 임베딩 모델 초기화, 상위 모델은 "text-embedding-3-large" 입니다.
embeddings_model = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model="text-embedding-3-small")

# PGVector Collection 이름 정의
COLLECTION_NAME_COMPANY = 'company_collection'
COLLECTION_NAME_NEWS = 'company_news_collection'
COLLECTION_NAME_UNIVERSITY = 'university_rank_collection'

# PGVector 스토어 인스턴스 를 위한 변수 생성
_company_vectorstore_instance = None
_news_vectorstore_instance = None
_university_vectorstore_instance = None

# 각 정보 별 Vectorstore 생성 함수
def get_company_vectorstore() -> PGVector:
	global _company_vectorstore_instance
	if _company_vectorstore_instance is None:
		logger.info(f"PGVector store collection 초기화 : {COLLECTION_NAME_COMPANY}")
		_company_vectorstore_instance =PGVector(
			collection_name = COLLECTION_NAME_COMPANY,
			connection_string = settings.DATABASE_URL,
			embedding_function = embeddings_model,
			use_jsonb = True,
			# JSONB 타입으로 메타데이터 필터링 성능 기대..
		)
	return _company_vectorstore_instance

def get_news_vectorstore() -> PGVector:
	global _news_vectorstore_instance
	if _news_vectorstore_instance is None:
		logger.info(f"PGVector store collection 초기화 : {COLLECTION_NAME_NEWS}")
		_news_vectorstore_instance = PGVector(
			collection_name = COLLECTION_NAME_NEWS,
			connection_string = settings.DATABASE_URL,
			embedding_function = embeddings_model,
			use_jsonb = True,
		)
	return _news_vectorstore_instance

def get_university_vectorstore() -> PGVector:
	global _university_vectorstore_instance
	if _university_vectorstore_instance is None:
		logger.info(f"PGVector store collection 초기화 : {COLLECTION_NAME_UNIVERSITY}")
		_university_vectorstore_instance = PGVector(
			collection_name = COLLECTION_NAME_UNIVERSITY,
			connection_string = settings.DATABASE_URL,
			embedding_function = embeddings_model,
			use_jsonb = True,
		)
	return _university_vectorstore_instance

# 각 정보 별 Retriever 생성 함수
def get_company_retriever(top_k: int = 4) -> BaseRetriever:
	"""회사 정보 검색을 위한 Retriever 생성"""
	# VectorStore 인스턴스 가져오기
	vectorstore = get_company_vectorstore()
	return vectorstore.as_retriever(search_kwargs={"k": top_k})    # 검색 결과 수 k 제어

def get_news_retriever(top_k: int = 4) -> BaseRetriever:
	"""뉴스 정보 검색을 위한 Retriever 생성"""
	vectorstore = get_news_vectorstore()
	return vectorstore.as_retriever(search_kwargs={"k": top_k})

def get_university_retriever(top_k : int = 1) -> BaseRetriever:
	"""대학 정보 검색을 위한 Retriever 생성"""
	vectorstore = get_university_vectorstore()
	return vectorstore.as_retriever(search_kwargs={"k": top_k})


# 문서 검색 로직 함수
async def retrieve_documents_from_sources(
	query: str, # company, company_news의 주 쿼리
	university_query: Optional[str] = None, # 대학 정보 검색 쿼리
	top_k_per_source: int = 4, # company, company_news 가져올 문서 수
	top_k_university: int = 1, # 대학 정보 가져올 문서 수
	) -> List[Document]:

	#검색된 모둔 문서 저장 리스트
	retrieved_docs = []

	# 대학 정보 검색
	try:
		university_retriever = get_university_retriever(top_k = top_k_university)

		# 비동기로 문서 검색
		university_docs = await university_retriever.aget_relevant_documents(university_query)

		if university_docs:
			logger.info(f"대학 정보 검색 결과 ({len(university_docs)})개")
			retrieved_docs.extend(university_docs)
		else:
			logger.info(f"쿼리 '{query}'에 대한 대학 정보 검색 결과 없음")
	except Exception as e:
		logger.error(f"대학 정보 검색 중 오류 발생: {e}")
	
	# 회사 정보 검색
	try:
		company_retriever = get_company_retriever(top_k=top_k_per_source)

		# 비동기로 문서 검색
		company_docs = await company_retriever.aget_relevant_documents(query)

		if company_docs:
			logger.info(f"회사 정보 검색 결과 ({len(company_docs)})개")
			retrieved_docs.extend(company_docs)
		else:
			logger.info(f"쿼리 '{query}'에 대한 회사 정보 검색 결과 없음")
	except Exception as e:
		logger.error(f"회사 정보 검색 중 오류 발생: {e}")
	
	# 뉴스 정보 검색
	try:
		news_retriever = get_news_retriever(top_k=top_k_per_source)

		# 비동기로 문서 검색
		news_docs = await news_retriever.aget_relevant_documents(query)

		if news_docs:
			logger.info(f"뉴스 정보 검색 결과 ({len(news_docs)})개")
			retrieved_docs.extend(news_docs)
		else:
			logger.info(f"쿼리 '{query}'에 대한 뉴스 정보 검색 결과 없음")
	except Exception as e:
		logger.error(f"뉴스 정보 검색 중 오류 발생: {e}")
	

	# 간단한 중복 제거
	unique_docs_dict: Dict[str, Document] = {}

	for doc in retrieved_docs:
		# page_content를 사용하여 고유성 보장
		if doc.page_content not in unique_docs_dict:
			unique_docs_dict[doc.page_content] = doc

	# 중복 제거된 최종 문서 
	final_docs = list(unique_docs_dict.values())
	logger.info(f"최종 검색 고유 문서 수: {len(final_docs)}")

	return final_docs

	