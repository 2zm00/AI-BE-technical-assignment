import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.core.config import settings

logger = logging.getLogger(__name__)

# LLM 인스턴스 생성
llm_instance : Optional[ChatOpenAI] = None

def get_llm_instance() -> ChatOpenAI:
	"""LLM 인스턴스 생성"""
	global llm_instance

	if llm_instance is None:
		if not settings.OPENAI_API_KEY:
			logger.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
			raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
		
		logger.info(f"ChatOpenAI 모델 {settings.OPENAI_MODEL_NAME} 초기화 중 ...")
		try:
			# LLM 인스턴스 세팅 정보
			llm_instance = ChatOpenAI(
				openai_api_key = settings.OPENAI_API_KEY,
				model_name = settings.OPENAI_MODEL_NAME,
				temperature = settings.OPENAI_TEMPERATURE,
			)
			logger.info("ChatOpenAI 모델 초기화 완료")

		except Exception as e:
			logger.error(f"ChatOpenAI 모델 초기화 실패: {e}", exc_info=True)
			raise

	return llm_instance


async def invoke_llm_for_experience(prompt: str) -> Optional[str]:
	"""
	주어진 프롬프트를 사용하여 LLM을 비동기로 호출하고 응답 텍스트 반환
	"""
	try:
		# LLM 인스턴스 가져오기
		llm = get_llm_instance()

		# LLM 전달할 메세지 리스트 생성
		messages = [
			HumanMessage(content=prompt)
		]

		logger.debug(f"LLM 전달 메세지 : {str(messages)[:300]}...")

		# LLM 비동기 호출
		response = await llm.ainvoke(messages)

		if response and hasattr(response, "content"):
			logger.debug(f"LLM 응답 : {str(response.content)[:300]}...")
			return response.content
		else:
			logger.warning("LLM 응답이 비어있거나 잘못된 형식입니다.")
			return None
	
	except Exception as e:
		logger.error(f"LLM 호출 중 오류 발생: {e}", exc_info=True)
		return None
