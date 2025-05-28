import logging
from typing import List
from fastapi import APIRouter, HTTPException, Body
from app.schemas.inference import TalentDataInput
from app.services.inference_service import infer_experiences_service

logger = logging.getLogger(__name__)

# API라우터 인스턴스 생성
router = APIRouter(
	prefix="/api/v1",
	tags=["Inference Service"],
)

@router.post(
	"/inference",
	response_model=List[str],
	summary="인재 데이터 추론 API",
	description="회사, 재직기간, 직무(타이틀) 만 존재하는 인재 데이터를 기반으로 LLM 을 활용하여 어떤 경험을 했는지, 어떤 역량을 가지고 있는지 추론하는 과제입니다.",
	response_description="추론된 경험 태그 리스트"
)

async def handle_infer_experience(
	talent_data: TalentDataInput = Body(
		...,
		examples={
			"talent_ex": {
				"summary": "talent_ex*.json의 형식을 따르는 예시입니다.",
				"lastName": "홍",
				"firstName": "길동",
				"positions": [
					{
						"title": "Chief Technology Officer",
						"companyLogo": "",
						"companyName": "엘박스",
						"description": "AI 솔루션 개발 총괄, 기술 전략 수립, 시리즈 B 투자 유치 지원",
						"startEndDate": {
							"start": {
								"year": 2023,
								"month": 3
							}
						},
						"companyLocation": "서울"
					},
				],
				 "educations": [
        {
            "grade": "",
            "degreeName": "석사",
            "schoolName": "서울대학교",
            "description": "",
            "fieldOfStudy": "컴퓨터공학",
            "startEndDate": "2010 - 2012",
            "originStartEndDate": {
                "endDateOn": {
                    "year": 2012
                },
                "startDateOn": {
                    "year": 2010
                }
            }
        }]
			}
		}
	)
): 
	"""
	JSON 기반으로 서비스 호출하고 결과를 반환합니다.
	"""

	try:
		logger.info(f"'/inference' API 요청 수신")
		inferred_experience_strings = await infer_experiences_service(talent_data)

		if not inferred_experience_strings and inferred_experience_strings is not None:
			logger.info("추론된 경험이 없거나 LLM 응답이 비어있습니다.")

		elif inferred_experience_strings is None:
			logger.error("None 반환했습니다.")
			raise HTTPException(status_code=500, detail="추론 중 내부 서버 오류 발생")
		
		logger.info(f"'/inference' API 응답 생성 완료")
		return inferred_experience_strings
	
	except HTTPException as http_exc:
		raise http_exc
	
	except ValueError as ve:
		logger.error(f"API 요청 처리 중 오류 발생: {ve}", exc_info=True)
		raise HTTPException(status_code=422, detail=f"요청 데이터 유효성 검사 실패: {str(ve)}")
	
	except Exception as e:
		logger.error(f"'/inference' API 처리 중 오류 발생: {e}", exc_info=True)
		raise HTTPException(status_code=500, detail="서버 오류 발생")