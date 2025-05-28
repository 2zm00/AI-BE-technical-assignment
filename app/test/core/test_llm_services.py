import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.core.llm_services import get_llm_instance, invoke_llm_for_experience
from app.core.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

# get_llm_instance 테스트
def test_get_llm_instance_succes(mocker):
	mocker.patch.object(settings, "OPENAI_API_KEY", "test_api_key")
	mocker.patch.object(settings, "OPENAI_MODEL_NAME", "gpt-test-model")
	mocker.patch.object(settings, "OPENAI_TEMPERATURE", 0.1 )

	mock_chat_open_ai_constructor = mocker.patch('app.core.llm_services.ChatOpenAI')
	mock_llm_obj = MagicMock(spec = ChatOpenAI)
	mock_chat_open_ai_constructor.return_value = mock_llm_obj

	# 처음 호출 시 초기화 되도록 None 설정
	mocker.patch('app.core.llm_services.llm_instance', None)

	# 언제
	llm1 = get_llm_instance()
	llm2 = get_llm_instance()

	mock_chat_open_ai_constructor.assert_called_once_with(
		openai_api_key = "test_api_key",
		model_name = "gpt-test-model",
		temperature = 0.1
	)

	assert llm1 is mock_llm_obj
	assert llm2 is mock_llm_obj

# invoke_llm_for_experience 테스트
@pytest.mark.asyncio
async def test_invoke_llm_success(mocker):
	mock_llm_obj = AsyncMock(spec=ChatOpenAI)
	mock_response_message = AIMessage(content="- 태그 (근거)")
	mock_llm_obj.ainvoke.return_value = mock_response_message

	# get_llm_instance 가 위에서 만든 Mock LLM 객체 반환
	mocker.patch('app.core.llm_services.get_llm_instance', return_value=mock_llm_obj)

	test_prompt = "테스트 프롬프트"

	#언제
	result = await invoke_llm_for_experience(test_prompt)

	assert result == "- 태그 (근거)"
	mock_llm_obj.ainvoke.assert_called_once()
	called_message = mock_llm_obj.ainvoke.call_args[0][0]
	assert len(called_message) == 1
	assert called_message[0].content == test_prompt

# API 요청 테스트
@pytest.mark.asyncio
async def test_invoke_llm_api_call_exception(mocker):
	mock_llm_obj = AsyncMock(spec=ChatOpenAI)
	mock_llm_obj.ainvoke.side_effect = Exception("Test API ERROR")
	mocker.patch('app.core.llm_services.get_llm_instance', return_value = mock_llm_obj)
	mock_logger_error = mocker.patch('app.core.llm_services.logger.error')

	# 언제
	result = await invoke_llm_for_experience("프롬프트")

	assert result is None
	mock_logger_error.assert_called_once()

# LLM 응답 없음 또는 invalid
@pytest.mark.asyncio
async def test_invoke_llm_empty_or_invalid_response(mocker):
	mock_llm_obj = AsyncMock(spec=ChatOpenAI)
	mock_llm_obj.ainvoke.return_value = AIMessage(content="")
	mocker.patch('app.core.llm_services.get_llm_instance', return_value=mock_llm_obj)
	mock_logger_warning = mocker.patch('app.core.llm_services.logger.warning')

	# 언제
	result_empty_content = await invoke_llm_for_experience("프롬프트1")

	assert result_empty_content == ""
	
	mock_llm_obj.ainvoke.return_value = MagicMock(spec=AIMessage)
	del mock_llm_obj.ainvoke.return_value.content

	result_no_content_attr = await invoke_llm_for_experience("프롬프트2")

	assert result_no_content_attr is None
	mock_logger_warning.assert_called_with("LLM 응답이 비어있거나 잘못된 형식입니다.")