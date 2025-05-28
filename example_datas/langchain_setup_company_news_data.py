import os
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any

from langchain_community.document_loaders import CSVLoader
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
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

COLLECTION_NAME_NEWS = "company_news_collection"

def load_news_data(file_path: str) -> List[Document]:
	"""뉴스 csv 불러오기"""
	docs = []
	try:
		with open(file_path, "r", encoding="utf-8") as file:
			reader = csv.DictReader(file)
			for i, row in enumerate(reader):
				try:
					company_name = row.get("name", "").strip()
					title = row.get("title", "").strip()
					original_link = row.get("original_link", "").strip()

					if not company_name or not title:
						logger.warning(f"행 {i+1}: 회사명 또는 제목 누락. 건너뜁니다.")
						continue
					
					year_str = row.get("year")
					month_str = row.get("month")
					day_str = row.get("day")
					news_date_str = "날짜 정보 기본값"

					if year_str and month_str and day_str:
						try:
							year_val = int(year_str)
							month_val = int(month_str)
							day_val = int(day_str)
							news_date = datetime(year_val, month_val, day_val)
							news_date_str = news_date.strftime("%Y-%m-%d")
						except ValueError as e:
							logger.warning(f"행 {i+1}: 날짜 변환 오류 ({e}). 건너뜁니다.")
							continue
					
					# 문서 객체 생성
					doc = Document(
						page_content=title,
						metadata={
							"company_name": company_name,
							"original_link": original_link,
							"news_date": news_date_str,
							"source_file": os.path.basename(file_path),
							"row_number": i + 2,
						},
					)
					docs.append(doc)
				
				except Exception as e:
					logger.error(f"CSV 행 처리 중 오류 발생 (row {i+1}): {e}, {row}")
					continue
			
			logger.info(f"'{file_path}' 파일에서 {len(docs)}개의 뉴스 Document를 성공적으로 로드했습니다.")
			return docs
		
	except FileNotFoundError:
		logger.error(f"파일을 찾을 수 없습니다: {file_path}")
		return []
	except Exception as e:
		logger.error(f"파일 로드 중 오류 발생: {e}")
		return []


def main():
	"""메인 함수"""

	#환경 변수 확인
	if not OPENAI_API_KEY:
		logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
		return
	if not DATABASE_URL.startswith("postgresql+psycopg2://"):
		logger.error("DATABASE_URL 환경 변수가 올바르지 않습니다.")
		return

	all_news_docs = []
	current_script_dir = os.path.dirname(os.path.abspath(__file__))
	news_file_path = os.path.join(current_script_dir, "company_news.csv")

	if not os.path.exists(news_file_path):
		logger.error(f"뉴스 데이터 파일을 찾을 수 없습니다: {news_file_path}")
		return
	
	logger.info(f"뉴스 데이터 파일 처리 시작: {news_file_path}")
	processed_news_docs = load_news_data(news_file_path)
	all_news_docs.extend(processed_news_docs)

	if not all_news_docs:
		logger.warning("임베딩하고 저장할 문서가 없습니다.")
		return

	logger.info(f"총 {len(all_news_docs)}개의 문서를 벡터 DB에 저장합니다.")

	try:
		db = PGVector.from_documents(
			documents=all_news_docs,
			embedding=embeddings_model,
			collection_name=COLLECTION_NAME_NEWS,
			connection_string=DATABASE_URL,
			pre_delete_collection=True,
			use_jsonb=True,
		)
		logger.info(f"'{COLLECTION_NAME_NEWS}' 컬렉션에 데이터 저장 완료")
	
	except Exception as e:
		logger.error(f"PGVector DB 저장 중 오류 발생: {e}")

if __name__ == "__main__":
	main()