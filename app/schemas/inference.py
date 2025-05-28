from pydantic import BaseModel, Field
from typing import Optional, List, Union, Any

"""공통 중첩 모델"""
class YearMonth(BaseModel):
	year: Optional[int] = Field(None, description="연도 (예: 2023)")
	month: Optional[int] = Field(None, description="월 (예: 5)")

class StartEndDate(BaseModel):
	start: Optional[YearMonth] = Field(None, description="시작일")
	end: Optional[YearMonth] = Field(None, description="종료일")

class EducationStartEndDate(BaseModel):
	startDateOn: Optional[YearMonth] = None
	endDateOn: Optional[YearMonth] = None


"""API 요청 바디 스키마"""
class Position(BaseModel):
	title: Optional[str] = Field(None, description="직무명")
	companyName: Optional[str] = Field(None, description="회사명")
	description: Optional[str] = Field(None, description="경력 상세 설명")
	startEndDate: Optional[StartEndDate] = Field(None, description="재직기간")
	companyLocation: Optional[str] = Field(None, description="회사 위치")

class Education(BaseModel):
	degreeName: Optional[str] = Field(None, description="학위명")
	schoolName: Optional[str] = Field(None, description="학교명")
	fieldOfStudy: Optional[str] = Field(None, description="전공")
	startEndDate: Optional[Union[str, EducationStartEndDate]] = Field(None, description="재학 기간")
	originStartEndDate: Optional[EducationStartEndDate] = None

class TalentDataInput(BaseModel):
	skills: Optional[List[str]] = Field(None, description="보유 기술")
	summary: Optional[str] = Field(None, description="개인 요약")
	website: Optional[List[str]] = Field(None, description="웹사이트 목록")
	headline: Optional[str] = Field(None, description="현재 직책/소속 요약")
	lastName: Optional[str] = Field(None, description="성")
	photoUrl: Optional[str] = Field(None, description="사진 URL")
	projects: Optional[List[Any]] = Field(None, description="프로젝트 목록")
	firstName: Optional[str] = Field(None, description="이름")
	positions: Optional[List[Position]] = Field(None, description="경력 목록")
	educations: Optional[List[Education]] = Field(None, description="학력 목록")
	linkedinUrl: Optional[str] = Field(None, description="LinkedIn URL")
	industryName: Optional[str] = Field(None, description="산업 분야")
	recommendations: Optional[List[Any]] = Field(None, description="추천서 목록")


# FastAPI에 표시할 모델 예시 값
class Config:
	json_schema_extra = {
		"example": {
                "skills": ["Large Language Models", "Python", "Machine Learning", "Deep Learning", "PyTorch"],
                "summary": "AI Researcher with experience in LLM development and deployment.",
                "headline": "AI Researcher at HyperConnect",
                "positions": [
                    {
                        "title": "AI Researcher",
                        "companyName": "HyperConnect",
                        "description": "Developing and researching large-scale language models for new services.\n- Pre-training and fine-tuning LLMs\n- Model optimization and serving",
                        "startEndDate": {"start": {"year": 2022, "month": 1}}
                    }
                ],
                "educations": [
                    {
                        "schoolName": "서울대학교",
                        "degreeName": "석사",
                        "fieldOfStudy": "인공지능",
                        "startEndDate": "2020 - 2022"
                    }
                ],
                "industryName": "AI / Machine Learning"
            }
	}
