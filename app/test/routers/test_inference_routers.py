import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.inference import TalentDataInput
from unittest.mock import patch, AsyncMock 

client = TestClient(app)

# Fixtures
@pytest.fixture
def valid_talent_payload() -> dict:
    # 완전한 인재 임시 데이터 정의
    return {
        "headline" : "테스트 잘하는 직업",
        "summary" : "테스트를 잘해보는 팀에 리드입니다",
        "skills" : ["AI", "Python", "Leadership"],
        "positions" : [
            {
                "companyName" : "Test Corp",
                "title" : "Sample title",
                "startEndDate" : {
                    "start" : {"year" : 2023, "month" : 3}
                }
            }
        ],
        "educations" : [
            {
                "schoolName" : "Test Univ",
                "degreeName" : "학사",
                "fieldOfStudy" : "테스트공학",
                "startEndDate" : "2010 - 2012",
                "originStartEndDate" : {
                    "startDateOn" : {"year" : 2010},
                    "endDateOn" : {"year" : 2012}
                }
            }
        ],
        "linkedinUrl" : "https://test.test.test",
        "industryname" : "테스트하기",
        "recommendations" : []
    }

@pytest.fixture
def invalid_talent_payload_missing_field() -> dict:
    # 오류 발생 유도하는 비어있는 인재 임시 데이터
    return {
        "skills": "not_a_list"
        } 



# --- 테스트 케이스 ---
def test_inference_endpoint_success_with_mock(mocker, valid_talent_payload: dict):
    # Given: infer_experiences_service가 특정 결과를 반환하도록 Mocking
    mocked_service_result = [
        "태그1 (근거1)",
        "태그2 (근거2)"
    ]
    # app.routers.inference 모듈 내에서 infer_experiences_service를 호출한다고 가정
    mock_infer_service = mocker.patch('app.routers.inference.infer_experiences_service', new_callable=AsyncMock, return_value=mocked_service_result)

    # When
    response = client.post("/api/v1/inference", json=valid_talent_payload)

    # Then
    assert response.status_code == 200
    assert response.json() == mocked_service_result
    mock_infer_service.assert_called_once()
    # 호출 시 전달된 TalentDataInput 객체 검증도 가능
    # call_args = mock_infer_service.call_args[0][0]
    # assert isinstance(call_args, TalentDataInput)
    # assert call_args.headline == valid_talent_payload["headline"]

def test_inference_endpoint_service_returns_empty(mocker, valid_talent_payload: dict):
    # Given: 서비스가 빈 리스트 반환
    mocker.patch('app.routers.inference.infer_experiences_service', new_callable=AsyncMock, return_value=[])
    
    # When
    response = client.post("/api/v1/inference", json=valid_talent_payload)

    # Then
    assert response.status_code == 200
    assert response.json() == [] # 빈 리스트 반환 확인

def test_inference_endpoint_service_returns_none_raises_500(mocker, valid_talent_payload: dict):
    # Given: 서비스가 None 반환 (내부 오류 상황 시뮬레이션)
    mocker.patch('app.routers.inference.infer_experiences_service', new_callable=AsyncMock, return_value=None)
    
    # When
    response = client.post("/api/v1/inference", json=valid_talent_payload)

    # Then
    assert response.status_code == 500
    assert response.json() == {"detail": "추론 중 내부 서버 오류 발생"} # 라우터의 에러 메시지 확인

def test_inference_endpoint_invalid_payload_type(invalid_talent_payload_missing_field: dict):
    # Given: 유효하지 않은 타입의 페이로드
    # When
    response = client.post("/api/v1/inference", json=invalid_talent_payload_missing_field)
    
    # Then
    assert response.status_code == 422 
    # response.json()['detail'] 등을 통해 구체적인 오류 메시지 확인 가능

def test_inference_endpoint_service_raises_value_error(mocker, valid_talent_payload: dict):
    # Given: 서비스 내부에서 ValueError 발생 시뮬레이션
    mocker.patch('app.routers.inference.infer_experiences_service', new_callable=AsyncMock, side_effect=ValueError("테스트용 값 오류"))

    # When
    response = client.post("/api/v1/inference", json=valid_talent_payload)

    # Then
    assert response.status_code == 422 # 라우터에서 ValueError를 422로 처리
    assert "요청 데이터 유효성 검사 실패: 테스트용 값 오류" in response.json()["detail"]

def test_inference_endpoint_service_raises_generic_exception(mocker, valid_talent_payload: dict):
    # Given: 서비스 내부에서 일반 Exception 발생 시뮬레이션
    mocker.patch('app.routers.inference.infer_experiences_service', new_callable=AsyncMock, side_effect=Exception("일반 서버 오류"))

    # When
    response = client.post("/api/v1/inference", json=valid_talent_payload)

    # Then
    assert response.status_code == 500
    assert response.json() == {"detail": "서버 오류 발생"}