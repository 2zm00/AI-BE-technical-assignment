from fastapi import FastAPI
import logging
from app.routers import inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="서치라이트 기술 과제 API",
    version="0.1.0",
    description="서치라이트 기술 과제입니다."
)

app.include_router(inference.router)

@app.get("/", tags=["Root"])
async def read_root():
    logger.info("루트 경로 '/' 수신")
    return {"message": ":서치라이트 기술 과제입니다. /docs 경로에서 API 문서를 확인해주십시오."}