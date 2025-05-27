<div align="center">
	<img width="15%" alt="SearchRight Logo" src="./public/assets/searchright_logo.webp">
</div>

# AI-BE-Technical-Assignment
> 서치라이트 에이아이 AI/BE 기술 과제

회사, 재직기간, 직무(타이틀) 만 존재하는 인재 데이터를 기반으로 LLM 을 활용하여 어떤 경험을 했는지, 어떤 역량을 가지고 있는지 추론하는 기술 과제

[![Repository](https://img.shields.io/badge/Repository-%23121011.svg?style=flat&logo=github&logoColor=white)](https://github.com/2zm00/AI-BE-technical-assignment)


## 설치 및 실행 방법

다음의 환경에서 준비해야합니다.
*   Python 3.11 이상
*   Poetry 
*   pyenv 
*   Docker 및 Docker Compose
*   PostgreSQL 데이터베이스 (pgvector 확장 활성화 필요)
*   OpenAI API Key

### 의존성 설치

```
git clone https://github.com/2zm00/AI-BE-technical-assignment
cd AI-BE-Technical-Assignment

$(poetry env activate) # 가상환경 설정 시
poetry install
```



프로젝트 루트의 `docker-compose.yaml` 파일을 사용하여 실행합니다.

```
# Docker Compose 컨테이너 빌드 및 실행
docker compose up -d
```


### 환경변수 설정
아래의 `.env.example`을 참고하여 다음 환경 변수를 설정합니다.
```
# 실제 사용할 값으로 사용합니다.

DATABASE_URL=postgresql+psycopg2://searchright:searchright@localhost:5432/searchright
OPENAI_API_KEY="sk-your_openai_api_key"
OPENAI_MODEL_NAME="gpt-4o" # 또는 사용하는 모델명
OPENAI_TEMPERATURE="0.1" # 일관된 결과 생성을 위한 권장 값
```

### 데이터베이스 테이블 생성 및 초기 데이터 주입
**`example_datas/` 폴더에 들어가 다음 스크립트들을 실행하여 Vector Store에 초기 데이터를 주입합니다.

1. **`example_datas`** 경로로 이동한 상태에서 아래 스크립트를 차례로 실행합니다.
1. `poetry run python langchain_setup_company_data.py`
2. `poetry run python langchain_setup_company_news_data.py`
3. `poetry run python langchain_setup_university_rank_data.py`

### API 서버 실행
**프로젝트 루트 디렉토리**에서 API 서버를 실행합니다.
```
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
`http://localhost:8000`에서 API 서버가 실행됩니다.

## 플로우 차트

### Langchain 컴포넌트 플로우차트
```mermaid
flowchart TD
    A["사용자: 인재 데이터 입력"] --> B["FastAPI Router: /api/v1/inference"]
    B --> C["inference_service.py"]
    C --> D1["**(일반/대학 쿼리 생성)** preprocess_talent_data_for_search_query "]
    D1 --> D2[" **(LangChain VectorStore)** get_company_vectorstore / get_news_vectorstore / get_university_vectorstore"]
    D2 --> D3["**(LangChain Retriever)** get_company_retriever / get_news_retriever / get_university_retriever "]
    D3 --> D4["**(문서 검색)** retrieve_documents_from_sources "]
    D4 --> E1["**(인재 프로필 포맷)** format_talent_profile_for_llm "]
    D4 --> E2["**(참고자료 포맷)** format_retrieved_documents_for_llm "]
    E1 & E2 --> F["LLM 프롬프트 조립"]
    F --> G["**(LangChain ChatOpenAI)** invoke_llm_for_experience"]
    G --> H["**(태그/근거 추출 및 정렬)** postprocess_llm_response "]
    H --> I["API 응답 반환"]
```

### 애플리케이션 플로우차트
```mermaid
flowchart LR
    A["클라이언트: POST /api/v1/inference"] --> B["FastAPI 라우터: /api/v1/inference"]
    B --> C{"infer_experiences_service"}

    subgraph GRAPH ["쿼리 생성"]
        direction LR
        C -- "인재 데이터" --> D["전처리"]
        D -- "일반 검색 쿼리" --> E["문서 검색 (Retrieval)"]

        C -- "인재 데이터.학력" --> F["대학명 추출"]
        F -- "대학 검색 쿼리" --> E
    end

    subgraph SG2 ["Vector DB "]
        direction TB
        style SG2 stroke:#333,stroke-width:2px
        G["대학 Retriever"]
        H["회사 Retriever"]
        I["뉴스 Retriever"]
    end

    E -- "대학 검색 쿼리" --> G
    E -- "일반 검색 쿼리" --> H
    E -- "일반 검색 쿼리" --> I

    G -- "검색된 대학 Docs" --> J["Docs 통합 및 중복 제거"]
    H -- "검색된 회사 Docs" --> J
    I -- "검색된 뉴스 Docs" --> J

    J -- "통합된 문서" --> K["컨텍스트 포맷팅"]

    C -- "인재 데이터" --> L["인재 프로필 포맷팅"]

    K -- "포맷팅된 컨텍스트" --> M["프롬프트 조립"]
    L -- "포맷팅된 인재 프로필" --> M

    M -- "최종 프롬프트" --> N["LLM 호출"]

    N -- "LLM raw 응답" --> O["응답 후처리"]

    O -- "후처리된 결과" --> P["결과 정렬"]

    P --> Q["최종 정렬된 결과"] --> B

```

## API

### API 호출
```
curl -X POST http://localhost:8000/api/v1/inference \
     -H "Content-Type: application/json" \
     -d @example_datas/talent_ex1.json 
```
또는 `/api/v1/inference` 를 통해 API를 호출할 수 있습니다.

### API 문서

본 프로젝트는 FastAPI를 사용하여 개발되었으며 다음 URL을 통해 API 명세를 제공하고 있습니다.
*   **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
    *   API를 테스트하고 요청/응답 스키마를 확인할 수 있습니다.
*   **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
    *   가독성 높은 형태로 API 문서를 제공합니다.

### 주요 엔드포인트

*   `POST /api/v1/inference`
    *   **설명:** 인재 데이터를 입력받아 경험 태그를 추론하여 반환합니다.
    *   **요청 본문:** `TalentDataInput` 스키마 (상세 내용은 Swagger/ReDoc 참조)
    *   **응답 본문:** 추론된 경험 태그 문자열 리스트 (`List[str]`)


## 디렉토리 구조

```
AI-BE-technical-assignment/
├── app(searchright-technical-assignment)/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 애플리케이션 인스턴스
│   ├── core/                      # --- 핵심 로직 및 설정 모듈 ---          
│   │   ├── __init__.py           
│   │   ├── config.py             # 환경 변수 및 애플리케이션 설정 관리
│   │   ├── llm_services.py       # LLM API 호출 관련 서비스
│   │   └── vector_db.py          # Vector DB 연결 및 검색 관련 서비스
│   ├── routers/                   # --- API 엔드포인트 정의 --- 
│   │   ├── __init__.py
│   │   └── inference.py          # '/api/v1/inference' 엔드포인트 로직
│   ├── schemas/                   # --- Pydantic 스키마 정의 ---
│   │   ├── __init__.py
│   │   └── inference.py          # 내부 데이터 구조 정의
│   ├── services/                  # --- 핵심 비즈니스 로직 구현 ---
│   │   ├── __init__.py
│   │   └── inference_service.py  # 인재 경험 추론 메인 서비스 로직
│   └── test/
│       ├── core/
│       │   ├── test_llm_services.py
│       │   └── test_vector_db.py
│       ├── routers/
│       │   └── test_inference_routers.py
│       ├── schemas/
│       │   └── test_inference_schemas.py
│       └── services/
│           └── test_inference_service.py
├── example_datas/ 
│   ├── langchain_setup_company_data.py
│   ├── langchain_setup_company_news_data.py
│   ├── langchain_setup_university_rank_data.py
│   ├── talent_ex1.json
│   ├── talent_ex2.json
│   ├── talent_ex3.json
│   └── talent_ex4.json
└── pyproject.toml

```

## 기술 스택

<div align="center">

| 영역         | 기술 스택                                                                 |
|--------------|--------------------------------------------------------------------------|
| **프로그래밍 언어** | Python 3.11 |
| **웹 프레임워크** | FastAPI |
| **LLM 연동** | LangChain, OpenAI API (`gpt-4o`, `text-embedding-3-small`) |
| **데이터베이스** | PostgreSQL, pgvector |
| **데이터 검증** | Pydantic |
| **의존성 관리** | Poetry |
| **테스트** | Pytest, Pytest-Mock, Pytest-Asyncio |
| **컨테이너** | Docker, Docker Compose |
| **형상 관리** | Git, Github |

</div>


## 테스트 실행 방법
본 프로젝트는 `pytest`를 사용하여 단위 테스트를 작성했습니다.
 테스트 실행 전, 개발 의존성이 설치되어 있는지 확인해주세요.

 **전체 테스트 실행**

```
poetry run pytest app/test/
```
    
 **특정 파일 테스트 실행**
```
poetry run pytest app/test/services/test_inference_service.py
```

**코드 커버리지 리포트 생성**
```
poetry run pytest --cov=app app/test/
poetry run coverage html  # HTML 리포트 생성 (htmlcov/index.html)
# 또는 poetry run coverage report # 콘솔에 요약 리포트 출력
```

## 향후 개선 사항

- **데이터 수집 및 업데이트 파이프라인 구축**
	
	- 현재 정적인 CSV 파일에서 로드하는 회사, 뉴스, 대학 순위 정보를 외부 데이터 소스와 연동하여 정보의 최신성과 정확성을 향상시킬 수 있습니다.
	- [KODATA](http://www.kodata.co.kr/), [혁신의 숲](https://www.innoforest.co.kr/), [DART](https://opendart.fss.or.kr/), [공공데이터포털](https://www.data.go.kr/) 등의 제공하는 서비스를 고려 할 수 있습니다.

- **LLM 성능 형상 및 모델 다양화**

	- 상위 LLM 모델과 임베딩 모델 (`gpt-4o` 이상, `text-embedding-3-large` )의 사용을 검토하고, 파인튜닝을 시도하여 추론 품질을 높일 수 있습니다.
	- 해당 프로젝트에서는 OpenAI API 토큰 리소스를 고려하여 `gpt-4o`, `text-embedding-3-small`을 사용하고 있습니다.

- **RAG 파이프라인 고도화**

	- 키워드 기반 검색과 벡터 검색을 결합한 하이브리드 검색도입을 더 정확한 컨텍스트를 제공할 수 있도록 검토할 수 있습니다.

- **사용자 인터페이스 개발**

	- 현재 API 형태로 제공되는 기능을 웹 기반의 사용자 인터페이스로 확장하여 개선할 수 있습니다.