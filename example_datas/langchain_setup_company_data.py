import os
import json
import glob
import logging
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://searchright:searchright@localhost:5432/searchright")

# Langchain 컴포넌트 초기화
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

COLLECTION_NAME_COMPANY = "company_collection"



def extract_texts_from_company_data(company_name: str, company_data: Dict[str, Any]) -> List[Document]:
	# company_ex 데이터에서 의미 있는 텍스트 추출하여 LangChain Document 객체 리스트로 반환
	docs = []

	# 1. 기본 회사 정보
	if company_data.get("base_company_info", {}).get("data", {}).get("seedCorp"):
		corp_info= company_data["base_company_info"]["data"]["seedCorp"]
		if corp_info.get("corpIntroKr"):
			docs.append(Document(
				page_content=f"회사소개: {corp_info['corpIntroKr']}",
				metadata={"company_name": company_name, "source": "corpIntroKr", "original_company_id": corp_info.get("id")}
			))
		if corp_info.get("bizInfoKr"):
			docs.append(Document(
				page_content=f"사업 분야: {corp_info['bizInfoKr']}",
				metadata={"company_name": company_name, "source": "bizInfoKr",
				"original_company_id" : corp_info.get("id")}
			))
	
	# 2. 특허 정보
	if company_data.get("patent", {}).get("list"):
		for patent_item in company_data["patent"]["list"]:
			if patent_item.get("title"):
				docs.append(Document(
					page_content=f"특허: {patent_item['title']}",
					metadata={"company_name": company_name, "source": "patent_title", "register_at": patent_item.get("registerAt")}
				))

	# 3. 제품 정보
	if company_data.get("products"):
		for product_item in company_data["products"]:
			if product_item.get("name"):
				docs.append(Document(
					page_content=f"주요 제품: {product_item['name']}",
					metadata={"company_name": company_name, "source": "product_name", "product_id": product_item.get("id")}
				))

	# 4. 회사 태그 정보
	if company_data.get("base_company_info", {}).get("data", {}).get("seedCorpTag"):
		tags = [tag.get("tagNameKr") for tag in company_data["base_company_info"]["data"]["seedCorpTag"] if tag.get("tagNameKr")]
		if tags:
			tags_text = ", ".join(tags)
			docs.append(Document(
				page_content=f"회사 관련 태그: {tags_text}",
				metadata={"company_name": company_name, "source": "company_tags"}
			))
	
	# 5. 재무 정보
	# if company_data.get("finance", {}).get("data", {}).get()

	logger.info(f"회사 '{company_name}에서 {len(docs)}개의 텍스트 조각 추출 완료.")
	return docs



def load_company_data(file_path: str) -> List[Document]:
	"""회사 파일 불러오기"""
	try:
		with open(file_path, "r", encoding="utf-8") as file:
			data = json.load(file)
		company_name = os.path.basename(file_path).split("_")[-1].split(".")[0]

		if data:
			extracted_docs = extract_texts_from_company_data(company_name, data)

			split_docs = text_splitter.split_documents(extracted_docs)
			logger.info(f"파일 '{file_path}'에서 {len(split_docs)}개의 분할된 문서 생성.")
			return split_docs
		else:
			logger.warning(f"파일 '{file_path}'에서 데이터 로드 실패.")
			return []
	except Exception as e:
		logger.error(f"파일 처리 중 오류 발생 ({file_path}): {e}")
		return []





def main():
	"""메인 함수"""

	# 환경 변수 확인
	if not OPENAI_API_KEY:
		logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
		return
	if not DATABASE_URL.startswith("postgresql+psycopg2://"):
		logger.error("DATABASE_URL 환경 변수가 올바르지 않습니다.")
		return

	all_docs_to_embed = []
	company_files = glob.glob(os.path.join("company_ex*.json"))

	logger.info(f"{len(company_files)}개의 회사 데이터 파일을 찾았습니다.")

	if not company_files:
		logger.warning("처리할 회사 데이터 파일이 없습니다.")
		return
	
	for file_path in sorted(company_files):
		logger.info(f"파일 처리 시작: {file_path}")
		processed_docs = load_company_data(file_path)
		all_docs_to_embed.extend(processed_docs)

	if not all_docs_to_embed:
		logger.warning("임베딩하고 저장할 문서가 없습니다.")
		return
	
	logger.info(f"총 {len(all_docs_to_embed)}개의 문서를 벡터 DB에 저장합니다.")

	try:
		db = PGVector.from_documents(
			documents=all_docs_to_embed,
			embedding=embeddings_model,
			collection_name=COLLECTION_NAME_COMPANY,
			connection_string=DATABASE_URL,
			# pre_delete_collection=True,
			use_jsonb=True,
		)
		logger.info(f"'{COLLECTION_NAME_COMPANY}' 컬렉션에 데이터 저장 완료")

	except Exception as e:
		logger.error(f"PGVector DB 저장 중 오류 발생: {e}")


if __name__ == "__main__":
	main()