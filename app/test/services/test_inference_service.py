import pytest
from unittest.mock import patch, AsyncMock
from typing import List
from langchain_core.documents import Document

from app.schemas.inference import TalentDataInput, Position, Education, StartEndDate, YearMonth, EducationStartEndDate
from app.services.inference_service import (
	extract_keywords_from_text,
	preprocess_talent_data_for_search_query,
	format_talent_profile_for_llm,
	format_retrieved_documents_for_llm,
	postprocess_llm_response,
	infer_experiences_service,
	#상수
	KEYWORDS,
)

# Fixture
@pytest.fixture
def sample_talent_data_for_service() -> TalentDataInput:
	return TalentDataInput(
		headline = "테스트 잘하는 책임자",
		summary = "테스트를 잘해보는 팀에 리드입니다",
		skills = ["AI", "Python", "Leadership"],
		positions = [
			Position(
				companyName = "Test Corp",
				description = "테스트 유치 합니다",
				startEndDate = StartEndDate(start= YearMonth(year = 2020, month =8), end =YearMonth(year=2023, month=2))
			)
		],
		education = [
			Education(
				schoolName = "Test Univ",
				degreeName = "학사",
				fieldOfStudy = "테스트공학",
				startEndDate = EducationStartEndDate(startDateOn=YearMonth(year=2010),endDateOn=YearMonth(year=2012)),
				originStartEndDate = EducationStartEndDate(startDateOn=YearMonth(year=2010), endDateOn=YearMonth(year=2012))
			)
		]
	)

@pytest.fixture
def sample_retrieved_docs() -> List[Document]:
	return [
		Document(page_content="Test Univ", metadata={"source_collection_name": "university_rank_collection", "university_name": "Test Univ", "rank": "0", "data_source": "중앙일보 평가 없음"}),
        Document(page_content="Test Corp, 테스트합니다", metadata={"source_collection_name": "company_news_collection", "company_name": "Test Corp", "news_date": "2023-05-01"}),
    ]

TARGET_EXPERIENCE_TAGS_FOR_TEST = [
		"물류 도메인 경험", "상위권 대학교", "대규모 회사 경험" ,"성장기 스타트업  경험", "리더쉽", "리더십", "대용량 데이터 처리 경험", "IPO", "M&A 경험", "신규 투자 유치 경험", "신기술 도입 경험", "글로벌 프로젝트 경험", "고객 관리 경험", "조직 관리 경험", "교육 및 멘토링 경험"
	]

DESIRED_TAG_ORDER_FOR_TEST = [
	"상위권 대학교", "대규모 회사 경험" ,"성장기 스타트업  경험", "리더쉽", "리더십", "대용량 데이터 처리 경험", "IPO", "M&A 경험", "신규 투자 유치 경험", "신기술 도입 경험", "글로벌 프로젝트 경험", "고객 관리 경험", "조직 관리 경험", "교육 및 멘토링 경험"
	]


# extract_keywords_from_text 테스트
def test_extract_keywords_found_and_not_found():
	text = "이 프로젝트는 IPO를 준비하고, M&A를 고려하며, 데이터 분석 및 AI 전략을 통해 성장했습니다. 리더십이 중요합니다."

	result = extract_keywords_from_text(text, KEYWORDS)
	assert "IPO" in result
	assert "M&A" in result
	assert "AI" in result

def test_extract_keywords_empty_inputs():
	assert extract_keywords_from_text("", KEYWORDS) == []
	assert extract_keywords_from_text("Some text", []) == []


#preprocess_talent_data_for_search_query 테스트
def test_preprocess_query_includes_career_skills_summary(sample_talent_data_for_service: TalentDataInput):
	query = preprocess_talent_data_for_search_query(sample_talent_data_for_service)
	assert "테스트 잘하는 책임자" in query
	assert "테스트를 잘해보는 팀에 리드입니다" in query
	assert "AI" in query
	assert "Test Corp" in query
	assert "테스트 유치 합니다" in query
	#학교 정보는 아직 포함되지 않아야함
	assert "Test Univ" not in query
	assert len(query) <= 1000

def test_preprocess_query_empty_talent_data():
	empty_talent = TalentDataInput()
	query = preprocess_talent_data_for_search_query(empty_talent)
	assert query == ""


# format_talent_profile_for_llm 테스트
def test_format_talent_profile_includes_all_sections(sample_talent_data_for_service: TalentDataInput):
	profile_str = format_talent_profile_for_llm(sample_talent_data_for_service)

	assert "### 학력 사항" in profile_str
	assert "(학력 정보 없음)" in profile_str
	assert "### 직무 요약" in profile_str
	assert "테스트 잘하는 책임자" in profile_str
	assert "### 개인 요약" in profile_str
	assert "테스트를 잘해보는 팀에 리드입니다" in profile_str
	assert "### 기술 요약" in profile_str
	assert "AI" in profile_str
	assert "### 경력 사항" in profile_str
	assert "Test Corp" in profile_str
	assert "테스트 유치 합니다" in profile_str

# format_retrieved_documents_for_llm 테스트
def test_format_retrieved_docs_structure_and_content(sample_retrieved_docs: List[Document]):
	formatted_str = format_retrieved_documents_for_llm(sample_retrieved_docs)
	assert "---참고 자료 시작---" in formatted_str
	assert "---참고 자료 끝---" in formatted_str
	assert "자료 1 출처: 알 수 없음, 회사명: 알 수 없음" in formatted_str
	assert "자료 2 출처: 알 수 없음, 회사명: Test Corp" in formatted_str

def test_format_retrieved_docs_empty_list():
	assert format_retrieved_documents_for_llm([]) == "검색된 결과가 없습니다."

# postprocess_llm_response 테스트
def test_postprocess_llm_response_valid_tags(sample_talent_data_for_service):
	llm_output = """
	- 상위권 대학교 (서울대학교, 국내 1위)
    - 리더십 (엘박스 CTO)
    - 없는태그 (이건무시)
    - M&A 경험 (요기요 매각 관련)
	"""

	#target_experience_tags 를 sample_talent_data_for_service 에서 가져옴
	target_tags = TARGET_EXPERIENCE_TAGS_FOR_TEST
	result = postprocess_llm_response(llm_output, target_tags)
	assert len(result) == 3
	assert "상위권 대학교 (서울대학교, 국내 1위)" in result
	assert "리더십 (엘박스 CTO)" in result
	assert "M&A 경험 (요기요 매각 관련)" in result

def test_postprocess_llm_response_tag_normalization():
	llm_output = "- 뤼더쉽 (팀장)"
	target_tags_leadership_only = ["리더십"]
	assert postprocess_llm_response(llm_output, target_tags_leadership_only) == []

def test_postprocess_llm_response_combined_tags():
	llm_output = "- IPO, M&A 경험 (엘박스 관련)"
	result = postprocess_llm_response(llm_output, TARGET_EXPERIENCE_TAGS_FOR_TEST)
	assert len(result) == 0

def test_postprocess_llm_response_empty():
	llm_output = ""
	result = postprocess_llm_response(llm_output, TARGET_EXPERIENCE_TAGS_FOR_TEST)
	assert len(result) == 0

# infer_experiences_service 테스트
@pytest.mark.asyncio
async def test_infer_experiences_service_end_to_end_mocked(mocker, sample_talent_data_for_service: TalentDataInput, sample_retrieved_docs: List[Document]):
	
	# 모든 외부 호출 모킹할 때
	mock_preprocess_query = mocker.patch('app.services.inference_service.preprocess_talent_data_for_search_query', return_value= "생성된 일반 쿼리")

	mock_retrieve_docs = mocker.patch('app.services.inference_service.retrieve_documents_from_sources',  new_callable = AsyncMock ,return_value=sample_retrieved_docs)

	mock_format_profile = mocker.patch('app.services.inference_service.format_talent_profile_for_llm', return_value="포매팅 인재 프로필")
	mock_format_context = mocker.patch('app.services.inference_service.format_retrieved_documents_for_llm', return_value="포매팅 참고자료")

	mocked_llm_raw_output = """
    - 상위권 대학교 (서울대학교, 중앙일보 평가 1위)
    - 리더십 (엘박스 CTO)
    - 신규 투자 유치 경험 (엘박스 시리즈 B)
    """

	mock_invoke_llm = mocker.patch('app.services.inference_service.invoke_llm_for_experience', new_callable=AsyncMock, return_value=mocked_llm_raw_output)

	result = await infer_experiences_service(sample_talent_data_for_service)

	mock_preprocess_query.assert_called_once_with(sample_talent_data_for_service)

	mock_retrieve_docs.assert_called_once_with(
		query = "생성된 일반 쿼리",
		university_query = None,
		top_k_per_source=4,
		top_k_university=1
	)

	mock_format_profile.assert_called_once_with(sample_talent_data_for_service)
	mock_format_context.assert_called_once_with(sample_retrieved_docs)

	mock_invoke_llm.assert_called_once()

	assert len(result) == 3
	assert "상위권 대학교 (서울대학교, 중앙일보 평가 1위)" in result
	assert "리더십 (엘박스 CTO)" in result
	assert "신규 투자 유치 경험 (엘박스 시리즈 B)" in result
