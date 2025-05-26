# 인재 데이터로부터 경험 태그를 추론하는 전체 프로세스를 서비스합니다.
# 주요 기능은 다음과 같습니다.
# 1. 입력된 인재 데이터로부터 VectorDB 검색을 위한 쿼리 생성 (전처리)
# 2. 인재 데이터를 LLM이 이해하기 쉬운 텍스트 형식으로 변환
# 3. VectorDB에서 관련 문서 검색
# 4. 검색된 문서를 LLM에 전달할 형식으로 변환
# 5. LLM 호출하여 경험 태그 추론 요청
# 6. LLM 응답 파싱 하여 최종 결과 형식으로 변환 (후처리)


import logging
from typing import List, Dict, Optional
from langchain_core.documents import Document
from app.core.vector_db import retrieve_documents_from_sources
from app.core.llm_services import invoke_llm_for_experience
from app.schemas.inference import TalentDataInput, StartEndDate, EducationStartEndDate, YearMonth


logger = logging.getLogger(__name__)


# 검색 쿼리 질 향상을 위해 주요 사용할 키워드 목록
# 이 키워드들을 통해 인재 설명에서 추출되어 검색 쿼리에 추가할 예정
KEYWORDS = ["투자", "리드", "Leadership", "Data", "Lead", "책임자", "대표", "IPO", "총괄", "인수", "합병", "CTO", "CEO", "CPO", "Head", "Manager", "Director", "성장", "유치", "전략", "분석", "운영", "AI", "M&A",]

# 키워드 단어 추출 함수
def extract_keywords_from_text(text: str, keywords: List[str]) -> List[str]:
	"""
	주어진 텍스트에서 정의된 키워드 목록에 있는 단어들을 추출합니다.
	"""
	if not text or not keywords:
		return []
	
	# 중복 방지 set
	found_keywords = set()
	# 대소문자 변환
	text_lower = text.lower()
	for kw in keywords:
		if kw.lower() in text_lower:
			found_keywords.add(kw)
	return list(found_keywords)


def preprocess_talent_data_for_search_query(talent_data: TalentDataInput) -> str:
	"""
	(전처리) 입력된 인재 데이터에서 벡터 DB 검색에 사용할 핵심 쿼리 생성
	"""

	query_parts = []

	# 1. 경력의 회사명과 직무를 주요 쿼리로 사용
	if talent_data.positions:
		for position in talent_data.positions:
			if position.companyName:
				query_parts.append(position.companyName)
			if position.title:
				query_parts.append(position.title)
			if position.description:
				# 설명의 첫 줄을 요약 정보로 활용
				first_line = position.description.split('\n')[0].strip()
				if first_line: query_parts.append(first_line[:100])
				# 설명에서 KEYWORD 추출
				keywords_in_desc = extract_keywords_from_text(position.description, KEYWORDS)
				if keywords_in_desc:
					query_parts.extend(keywords_in_desc)
				
	# 2. 기술/헤드라인/요약 정보 추가
	if talent_data.skills:
		query_parts.append(", ".join(talent_data.skills))
	if talent_data.headline:
		query_parts.append(talent_data.headline)
	if talent_data.summary:
		query_parts.append(talent_data.summary[:100])
	
	
	# 3. 쿼리 조각 정리 및 결합
	processed_query_parts = []
	
	for part in query_parts:
		if isinstance(part, list):
			# 각 요소 문자열 반환 및 공백 제거
			processed_query_parts.extend([str(p).strip() for p in part if p])
		elif isinstance(part, str):
			processed_query_parts.append(part.strip())

	# 문자열 공백 필터링 부분
	filtered_parts = list(dict.fromkeys(filter(None, processed_query_parts)))

	search_query = " ".join(filtered_parts)

	max_query_length = 5000
	if len(search_query) > max_query_length:
		logger.warning(f"생성된 검색 쿼리가 너무 깁니다 (길이: {len(search_query)}). 일부를 잘라냅니다.")
		search_query = search_query[:max_query_length]
	
	if not search_query:
		logger.error("벡터 DB 검색 쿼리가 비어있습니다.")
		return ""
	
	logger.info(f"생성된 벡터 DB 검색 쿼리 : '{search_query}'")
	return search_query


# 데이터 - 텍스트 변환
def format_talent_profile_for_llm(talent_data: TalentDataInput) -> str:
	"""
	인재 데이터를 LLM이 이해하기 좋은 텍스트 형식으로 변환
	경력 기간, 학력 기간등의 날짜 정보 포함
	"""
	parts=[]

	# 학력 사항 포맷팅
	parts.append("\n### 학력 사항")
	if talent_data.educations:
		for i, edu in enumerate(talent_data.educations):
			edu_period = "기간 정보 없음"
			
			# 학력의 originStartEndDate 우선적으로 사용
			if hasattr(edu, 'originStartEndDate') and edu.originStartEndDate and hasattr(edu.originStartEndDate, 'startDateOn') and hasattr(edu.originStartEndDate, 'endDateOn'):
				start_obj = edu.originStartEndDate.startDateOn
				end_obj = edu.originStartEndDate.endDateOn
				# 연도.월 형식으로 시작일 포매팅
				# 입학 연도
				start_year_str = str(start_obj.year) if start_obj and hasattr(start_obj, 'year') and start_obj.year is not None else ""
				# 입학 월
				if start_obj and hasattr(start_obj, 'month') and start_obj.month is not None: start_year_str += f".{start_obj.month:02d}"
				# 졸업 연도
				end_year_str = str(end_obj.year) if end_obj and hasattr(end_obj, 'year') and end_obj.year is not None else ""
				# 졸업 월
				if end_obj and hasattr(end_obj, 'month') and end_obj.month is not None: end_year_str += f".{end_obj.month:02d}"

				# 학력 포매팅
				if start_year_str and end_year_str: edu_period = f"{start_year_str} ~ {end_year_str}"
				elif start_year_str: edu_period = f"{start_year_str} ~ 현재"

			# 학력의 startEndDate 사용
			elif edu.startEndDate:
				# 기간이 문자열인 경우
				if isinstance(edu.startEndDate, str):
					edu_period = edu.startEndDate
				elif edu.startEndDate and hasattr(edu.startEndDate, 'startDateOn') and hasattr(edu.startEndDate, 'endDateOn'):
					start_edu_obj = edu.startEndDate.startDateOn
					end_edu_obj = edu.startEndDate.endDateOn
					# 입학 연도
					start_year_str = str(start_edu_obj.year) if start_edu_obj and hasattr(start_edu_obj, 'year') and start_edu_obj.year is not None else ""
					# 입학 월
					if start_edu_obj and hasattr(start_edu_obj, 'month') and start_edu_obj.month is not None: start_year_str += f".{start_edu_obj.month:02d}"
					# 졸업 연도
					end_year_str = str(end_edu_obj.year) if end_edu_obj and hasattr(end_edu_obj, 'year') and end_edu_obj.year is not None else ""
					# 졸업 월
					if end_edu_obj and hasattr(end_edu_obj, 'month') and end_edu_obj.month is not None:
						end_year_str += f".{end_edu_obj.month:02d}"

					# 학력 포매팅
					if start_year_str and end_year_str:
						edu_period = f"{start_year_str} ~ {end_year_str}"
					elif start_year_str: 
						edu_period = f"{start_year_str} ~ 현재"
			# 학위, 전공 포매팅
			degree_field_str = f"{edu.degreeName or ''} {edu.fieldOfStudy or ''}".strip()
			# 학력 포매팅
			education_line = f"{i+1}. {edu.schoolName or '학교명 미기재'} - {degree_field_str or '학위/전공 미기재'} ({edu_period})"
			parts.append(education_line)
	else:
		parts.append("  (학력 정보 없음)")

	# 직무, 개인, 기술 요약 정보 포매팅
	# 직무 요약
	if talent_data.headline:
		parts.append(f"### 직무 요약 : \n{talent_data.headline.strip()}")
	# 개인 요약
	if talent_data.summary:
		parts.append(f"### 개인 요약 : \n{talent_data.summary.strip()}")
	# 기술 요약
	if talent_data.skills:
		parts.append(f"### 기술 요약 : \n{", ".join(talent_data.skills)}")

	# 경력사항 포매팅
	parts.append("\n### 경력 사항:")
	if talent_data.positions:
		for i, pos in enumerate(talent_data.positions):
			start_str = "시작일 정보 없음"
			end_str = "현재"

			# 경력 시작일, 종료일 파싱
			if pos.startEndDate:
				# 경력 시작 연도
				if pos.startEndDate.start and hasattr(pos.startEndDate.start, 'year') and pos.startEndDate.start.year is not None:
					start_str = str(pos.startEndDate.start.year)
					# 경력 시작 월
					if hasattr(pos.startEndDate.start, 'month') and pos.startEndDate.start.month is not None:
						start_str += f".{pos.startEndDate.start.month:02d}"

				# 경력 종료 연도
				if pos.startEndDate.end and hasattr(pos.startEndDate.end, 'year') and pos.startEndDate.end.year is not None:
					end_str = str(pos.startEndDate.end.year)
					# 경력 종료 월
					if hasattr(pos.startEndDate.end, 'month') and pos.startEndDate.end.month is not None:
						end_str += f".{pos.startEndDate.end.month:02d}"

			position_line = f"{i+1}. {pos.companyName or '회사명 미기재'} - {pos.title or '직무 미기재'} ({start_str} ~ {end_str})"
			
			#직무 설명 포매팅
			if pos.description:
				desc_text = pos.description.strip().replace("\n", "\n    ")
				position_line += f"\n    주요 업무/성과: \n    {desc_text}"
				parts.append(position_line)
	else:
		parts.append("\n (경력정보없음)")
			
	
	return "\n".join(parts)


# 문서 전달 포매팅
def format_retrieved_documents_for_llm(documents: List[Document], max_chars_per_doc: int=500, max_total_context_chars: int=10000) -> str:
	"""
	(문서 전달)검색된 Document 객체를 LLM에 전달시키기 위하여 변환
	"""

	# 문서가 없을 시 
	if not documents:
		return "검색된 결과가 없습니다."
	
	# context 시작 문자
	context_str= "---참고 자료 시작---\n"

	for i, doc in enumerate(documents):
		# 문서 내용 미리보기
		content_preview = doc.page_content.replace("\n", " ").strip()[:max_chars_per_doc]

		# 메타데이터 추출, 포맷팅
		source = doc.metadata.get('source', '알 수 없음')
		company_name_md = doc.metadata.get('company_name', '알 수 없음')
		news_date_md = doc.metadata.get('news_date', '')
		
		metadata_info = f"출처: {source}"
		if company_name_md != '정보 없음': 
			metadata_info += f", 회사명: {company_name_md}"
		if news_date_md: 
			metadata_info += f", 날짜: {news_date_md}"

		context_str += f"자료 {i+1} {metadata_info}: {content_preview}\n"

	# context 종료 문자
	context_str += "---참고 자료 끝---"
	return context_str


# 후처리 형식 변환
def postprocess_llm_response(llm_output: Optional[str], target_experience_tags: List[str]) -> List[str]:
	"""
	(후처리) LLM의 텍스트 응답을 파싱하여 json output 형식으로 변환
	"""
	if not llm_output:
		logger.warning("LLM 응답이 비어있습니다.")
		return []
	
	processed_results: List[str] = []
	lines = llm_output.strip().split('\n') # 응답 줄 단위로 분리


	for line in lines:
		line_content_original = line.strip() # 응답 원본 내용
		line_content = line_content_original

		# 각 줄이 "-"로 시작하는 경우, 해당 부분을 제거
		if line_content.startswith("- "):
			line_content = line_content[2:].strip()

		if not line_content: 
			continue

		tag_part_final = ""
		evidence_part_final = "근거 명시 안됨"

		# 근거 형식 포맷팅
		if "(" in line_content and line_content.endswith(")"):
			try:
				tag_candidate, evidence_candidate = line_content.rsplit("(", 1)
				tag_candidate = tag_candidate.strip()
				evidence_candidate = evidence_candidate[:-1].strip() 

				# 정규화된 태그명으로 목표 태그 목록과 비교
				normalized_tag_candidate = tag_candidate.lower().replace(" ", "")

				for target_tag in target_experience_tags:
					normalized_target_tag = target_tag.lower().replace(" ", "")
					if normalized_tag_candidate == normalized_target_tag:
						tag_part_final = target_tag # 목표 태그명

						if evidence_candidate:
							evidence_part_final = evidence_candidate
						break
				
				# 일치하는 목표 태그가 없을 경우
				if not tag_part_final:
					logger.warning(f"인식된 태그가 경험 태그 목록에 없습니다: {tag_candidate}")

			except ValueError:
				logger.warning(f"LLM 응답에서 괄호 처리 오류: {line_content_original}")
				tag_candidate_only = line_content_original.split(" (")[0].strip()

				normalized_tag_candidate_only = tag_candidate_only.lower().replace(" ", "")
				for target_tag in target_experience_tags:
					if target_tag.lower().replace(" ", "") == normalized_tag_candidate_only:
						tag_part_final = target_tag
						break
				if not tag_part_final:
					logger.warning(f"인식된 태그가 경험 태그 목록에 없습니다: {line_content_original}")

		# 태그만 있는 경우
		elif line_content:

			tag_candidate_only = line_content.strip()
			normalized_tag_candidate_only = tag_candidate_only.lower().replace(" ", "")

			for target_tag in target_experience_tags:
				normalized_target_tag = target_tag.lower().replace(" ", "")
				if normalized_tag_candidate_only == normalized_target_tag:
					tag_part_final = target_tag
					break
			
			# 근거 없이 반환한 태그 
			if not tag_part_final:
				logger.warning(f"LLM이 근거 없이 반환한 태그 '{tag_candidate_only}'는 목표 태그 목록에 없습니다. 해당 라인 무시: '{line_content}'")
		
		# 최종적으로 유효 태그 식별 시 결과에 추가
		if tag_part_final:
			logger.debug(f"  -> 성공적으로 파싱됨: 태그='{tag_part_final}', 근거='{evidence_part_final}'")
			# 결과 추가
			processed_results.append(f"{tag_part_final} ({evidence_part_final})")


	logger.info(f"LLM 응답 후처리 결과 (항목 수 : {len(processed_results)}) : {processed_results}")
	return processed_results

# 메인 추론 서비스  함수
async def infer_experiences_service(talent_data: TalentDataInput) -> List[str]:
	"""
	인재 데이터에 대한 경험 태그를 추론하는 서비스
	"""

	# 벡터 DB에서 검색 쿼리 생성
	search_query = preprocess_talent_data_for_search_query(talent_data)

	# 별도로 대학교 쿼리 생성
	university_query_str: Optional[str] = None

	# 학력 정보 추가
	if talent_data.educations:
		for edu in talent_data.educations:
			if edu.schoolName and edu.schoolName.strip():
				university_query_str = edu.schoolName.strip()
				logger.info(f"대학 검색을 위한 학교 명 : {university_query_str}")
				break


	# 벡터 DB에서 문서 검색
	retrieved_docs: List[Document] = []

	# search_query, university_query_str 중 하나라도 존재해야 검색 시도
	if search_query or university_query_str:
		try:
			# university_query_str 만 있을 경우
			retrieved_docs = await retrieve_documents_from_sources(query= search_query if search_query else "정보없음", university_query=university_query_str, top_k_per_source=4, top_k_university=1)
	
		except Exception as e:
			logger.error(f"문서 검색 단계 예외 발생: {e}", exc_info = True)

	# LLM에 전달할 포맷으로 변환
	formatted_context = format_retrieved_documents_for_llm(retrieved_docs)

	# 경험 태그 목록
	target_experience_tags_for_prompt = [
		"물류 도메인 경험", "상위권 대학교", "대규모 회사 경험" ,"성장기 스타트업  경험", "리더쉽", "리더십", "대용량 데이터 처리 경험", "IPO", "M&A 경험", "신규 투자 유치 경험", "신기술 도입 경험", "글로벌 프로젝트 경험", "고객 관리 경험", "조직 관리 경험", "교육 및 멘토링 경험"
	]

	# 프롬프트에 넣을 경험 태그 변수
	target_tags_str_for_prompt = ", ".join(target_experience_tags_for_prompt)

	# 인재 프로필 요약 (LLM에게 전달하기 위해)
	talent_profile_summary_parts = []

	# 요약 정보 추가
	if talent_data.headline:
		talent_profile_summary_parts.append(talent_data.headline)
	if talent_data.summary:
		talent_profile_summary_parts.append(talent_data.summary)
	if talent_data.skills:
		talent_profile_summary_parts.append(f"보유 기술 : {','.join(talent_data.skills)}")
	if talent_data.positions:
		talent_profile_summary_parts.append("\n 경력 사항:")
		# 회사 근무 경력 정보
		for i, pos in enumerate(talent_data.positions):
			start_str = "시작일 정보 없음"
			end_str = "종료일 정보 없음"

			if pos.startEndDate and pos.startEndDate.start:
				start_str = str(pos.startEndDate.start.year)
				if pos.startEndDate.start.month:
					start_str += f".{pos.startEndDate.start.month:02d}"
					end_str = "현재"
			if pos.startEndDate and pos.startEndDate.end and pos.startEndDate.end.year: 
				end_str = str(pos.startEndDate.end.year)
				if pos.startEndDate.end.month:
					end_str += f".{pos.startEndDate.end.month:02d}"
			# 회사, 직책, 설명 정보
			position_info = f"{pos.companyName} - {pos.title}"
			if pos.description:
				desc = "\n   ".join([line.strip() for line in pos.description.strip().split('\n') if line.strip()])
				desc = desc[:400] + "..." if len(desc) > 400 else desc
				position_info += f"\n 경력 설명 : {desc}"
			talent_profile_summary_parts.append(position_info)
	else:
		talent_profile_summary_parts.append("\n 경력 사항:\n (경력 정보 없음)")

	# 대학교 쿼리에 포함될 수 있도록
	if talent_data.educations:
		talent_profile_summary_parts.append("\n 학력 사항:")
		university_names = [edu.schoolName.strip() for edu in talent_data.educations if edu.schoolName]
		if university_names:
			university_query_str = " ".join(university_names)
	else:
		talent_profile_summary_parts.append("\n 학력 사항:\n (학력 정보 없음)")

	talent_profile_for_llm = format_talent_profile_for_llm(talent_data)
	logging.info(f"llm에게 전달하는 talent_data: \n{talent_profile_for_llm}")


	# LLM에 전달할 프롬프트 조립
	prompt = f"""

당신은 고도로 숙련된 HR 전문가이자 정교한 경력 분석가입니다. 
당신의 주요 목표는 제공된 인재 프로필과 참고 자료를 **종합적으로 분석**하여, 사전에 정의된 '경험 태그 목록'에 해당하는 **모든 경험을 빠짐없이 식별**하고, 각 경험에 대한 **명확하고 타당한 근거를 제시**하는 것입니다.

--- 인재 프로필 시작 ---
{talent_profile_for_llm}
--- 인재 프로필 끝 ---

{formatted_context}

지시사항:
1.  아래 '경험 태그 목록'에 있는 **각 태그에 대해 개별적으로 해당 여부를 판단**하고, 해당하는 경우 **목록에 있는 정확한 태그명만을 사용**하여 경험을 **전부** 식별하십시오. **절대로 여러 태그를 하나로 합치거나(예: 'IPO, M&A 경험'과 같이 쉼표로 연결 금지), 태그명을 변형하거나, 목록에 없는 새로운 태그를 만들지 마십시오. 태그를 변형하거나 합치는 행위는 금지됩니다.**
2.  선택된 각 경험에 대해, 판단의 근거가 되는 구체적인 내용(예: 회사명, 프로젝트명, 성과, 기술 스택, 학교명, **근무 기간, 관련 이벤트 발생 시점** 등)을 **반드시 인재 프로필이나 참고 자료에서 찾아** 간략하게 괄호 안에 명시해주십시오. 특히, "IPO", "M&A 경험", "신규 투자 유치 경험" 태그는 **인재의 재직 기간과 이벤트 발생 시점이 일치하거나 밀접하게 연관되어야 하며, 이 시간적 연관성을 근거에 명시**해주십시오.
3.  최종 결과는 각 경험과 근거를 **"- 경험 태그명 (근거)" 형식으로 한 줄에 하나씩 나열**해야 합니다. **각 줄에는 정확히 하나의 태그명만 포함**되어야 하며, 각 항목은 '-'로 시작해주십시오. 태그명은 '경험 태그 목록'의 항목과 **완전히 동일하게 작성**해야 합니다. (아래 '추론된 경험 목록 예시' 참고)
4.  근거는 최대한 간결하고 핵심적인 내용만 포함시켜 주십시오. 만약 여러 근거가 있다면 가장 대표적인 것을 언급하거나 요약해주십시오.
5.  '경험 태그 목록'에 없는 경험은 생성하지 마십시오.
6.  만약 특정 경험 태그에 대한 명확한 근거를 찾기 어렵지만 강하게 추정된다면, 근거 부분에 '(추정 근거: [인재 프로필의 어떤 내용 또는 참고 자료의 어떤 정보 때문에 추정하는지에 대한 간략한 이유])'와 같이 **구체적인 추정 이유**를 명시해주십시오. (단순 '(추정)'만으로는 부족합니다.)
7.  **"상위권대학교" 태그 생성 규칙 (매우 중요, 가장 우선적으로 판단하십시오):**
    a.  인재 프로필의 학력 사항에 기재된 각 학교명(예: '서울대학교', '연세대학교')을 면밀히 확인합니다.
    b.  '---참고 자료 시작---'과 '---참고 자료 끝---' 사이에 해당 학교명과 관련된 대학 순위 정보(예: '자료 X ... 대학명: 서울대학교, 순위: 1위 (출처: 중앙일보 2024년 평가)')가 있는지 찾아보십시오. **참고 자료에 있는 대학 순위 정보는 매우 중요한 판단 근거입니다.**
    c.  **참고 자료에서 해당 학교가 명시적으로 상위권(예: 국내 대학 평가 1위~20위 이내)으로 확인되면, 반드시 "- 상위권대학교 (학교명, [참고 자료에 명시된 순위 및 출처 정보])" 형식으로 태그를 생성하십시오.** (예: "- 상위권대학교 (서울대학교, 중앙일보 2024년 평가 1위)")
    d.  참고 자료에 해당 학교 정보가 없거나 순위 정보가 명확하지 않더라도, 해당 학교가 **대한민국 내에서 일반적으로 최상위 명문 대학(예: 서울대학교, 연세대학교, 고려대학교, KAIST, POSTECH 등 누구나 인정하는 수준의 대학)으로 널리 알려져 있다면, "- 상위권대학교 (학교명, 일반적인 사회적 인지도 기반)" 형식으로 태그를 생성**하십시오.
    e.  해외 대학의 경우, 세계적으로 인정받는 최상위권 대학(예: MIT, Stanford, Harvard 등)이거나 참고 자료에서 명확한 상위권 근거가 있을 때만 "상위권대학교" 태그를 생성하고, 그 외 해외 대학은 이 태그를 생성하지 마십시오.
    f.  'OO대'와 'OO대학교'는 동일하게 취급하여 판단하십시오.

경험 태그 목록: {target_tags_str_for_prompt}

추론된 경험 목록 예시 **(아래는 다양한 상황에 대한 예시이며, 실제 응답은 인재 프로필과 참고 자료에 따라 달라져야 합니다. 형식과 태그명 사용 방식을 주의 깊게 보십시오.)**: 
- 상위권대학교 (서울대학교, 중앙일보 2024년 평가 1위)
- 대규모 회사 경험 (네이버 재직 중, 직원 수 5,000명 이상)
- 성장기 스타트업 경험 (토스 재직 시, 시리즈 C 투자 유치 및 조직 3배 성장 기여)
- 리더십 (엘박스 CTO, 개발팀 20명 총괄)
- IPO (밀리의서재 CFO 재직 중, 2023년 코스닥 상장 성공)
- M&A 경험 (요기요 재직 중, 2021년 컴바인드딜리버리-딜리버리히어로 M&A 기술 실사 참여)
- 신규 투자 유치 경험 (스타트업X 시리즈 A 투자 유치 IR 자료 작성 및 발표, 2022년)
- 대용량 데이터 처리 및 분석 (빅데이터 플랫폼 Y 구축 프로젝트 참여, 일일 1TB 데이터 처리)

추론된 경험 목록:
"""
	
	# LLM 호출 (비동기)
	llm_raw_response = await invoke_llm_for_experience(prompt)

	if llm_raw_response is None:
		logger.warning("LLM 응답이 없습니다.")
		return []
	
	logger.info(f"LLM 원본 응답 수신 : {llm_raw_response}")

	# LLM 응답 후처리
	final_output_strings = postprocess_llm_response(llm_raw_response, target_experience_tags_for_prompt)

	
	# 태그를 원하는 순서로 정렬하기 위한 기준 리스트
	DESIRED_TAG_ORDER = ["상위권 대학교", "대규모 회사 경험" ,"성장기 스타트업  경험", "리더쉽", "리더십", "대용량 데이터 처리 경험", "IPO", "M&A 경험", "신규 투자 유치 경험", "신기술 도입 경험", "글로벌 프로젝트 경험", "고객 관리 경험", "조직 관리 경험", "교육 및 멘토링 경험"]

	def get_tag_from_final_string(result_str: str) -> Optional[str]:
		"""
		LLM 응답에서 태그 부분을 추출하는 헬퍼 함수
		"""
		if result_str and "(" in result_str:
			return result_str.split(" (", 1)[0].strip()
		elif result_str:
			return result_str.strip()
		return None
	
	def sort_key_for_tags(result_str: str):
		"""
		태그 원하는 순서로 정렬하는 함수
		"""
		tag = get_tag_from_final_string(result_str)
		if tag and tag in DESIRED_TAG_ORDER:
			try:
				return DESIRED_TAG_ORDER.index(tag)
			except ValueError:
				return len(DESIRED_TAG_ORDER)
		return len(DESIRED_TAG_ORDER)
	
	# 최종 태그 결과 정렬
	final_sorted_output_strings = sorted(final_output_strings, key=sort_key_for_tags)

	logger.info(f"최종 출력 결과 : {final_sorted_output_strings}")

	return final_sorted_output_strings