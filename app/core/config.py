from pydantic_settings import BaseSettings, SettingsConfigDict

# OPENAI 세팅값
class Settings(BaseSettings):
	OPENAI_API_KEY: str
	DATABASE_URL: str
	# 토큰 비용 고려하여 4o 모델 사용, 리소스 여유가 있다면 높은 모델 사용 고려
	OPENAI_MODEL_NAME: str = "gpt-4o"
	# TEMPERATURE 값 0.0 ~ 0.3 사이로 설정, 정확한 결과를 위해 0.0으로 우선 사용함.
	OPENAI_TEMPERATURE: float = 0.0
	model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()