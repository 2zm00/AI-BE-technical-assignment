# test/core/test_vector_db.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.vector_db import (
    #함수
    get_company_vectorstore,
    get_news_vectorstore, 
    get_university_vectorstore,
    get_company_retriever,
    get_news_retriever,
    get_university_retriever,
    retrieve_documents_from_sources,
    #상수
    COLLECTION_NAME_COMPANY, COLLECTION_NAME_NEWS, COLLECTION_NAME_UNIVERSITY, embeddings_model
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import PGVector
from app.core.config import settings


# Fixtures
@pytest.fixture
def mock_pgvector_constructor(mocker):
    return mocker.patch('app.core.vector_db.PGVector')

@pytest.fixture(autouse=True)
def reset_vectorstore_instances(mocker):
    mocker.patch('app.core.vector_db._company_vectorstore_instance',None)
    mocker.patch('app.core.vector_db._news_vectorstore_instance',None)
    mocker.patch('app.core.vector_db._university_vectorstore_instance',None)

# get vectorstore 테스트
def test_get_company_vectorstore_initialization(mock_pgvector_constructor: MagicMock):
    mock_instance = MagicMock(spec=PGVector)
    mock_pgvector_constructor.return_value = mock_instance

    vs1 = get_company_vectorstore()
    vs2 = get_company_vectorstore()

    mock_pgvector_constructor.assert_called_once_with(
        collection_name = COLLECTION_NAME_COMPANY,
        connection_string=settings.DATABASE_URL,
        embedding_function=embeddings_model,
        use_jsonb=True
    )

    assert vs1 is mock_instance
    assert vs2 is mock_instance

def test_get_news_vectorstore_initialization(mock_pgvector_constructor: MagicMock):
    mock_instance = MagicMock(spec=PGVector)
    mock_pgvector_constructor.return_value = mock_instance

    vs1 = get_news_vectorstore()
    vs2 = get_news_vectorstore()

    mock_pgvector_constructor.assert_called_once_with(
        collection_name = COLLECTION_NAME_NEWS,
        connection_string=settings.DATABASE_URL,
        embedding_function=embeddings_model,
        use_jsonb=True
    )

    assert vs1 is mock_instance
    assert vs2 is mock_instance

def test_get_university_vectorstore_initialization(mock_pgvector_constructor: MagicMock):
    mock_instance = MagicMock(spec=PGVector)
    mock_pgvector_constructor.return_value = mock_instance

    vs1 = get_university_vectorstore()
    vs2 = get_university_vectorstore()

    mock_pgvector_constructor.assert_called_once_with(
        collection_name = COLLECTION_NAME_UNIVERSITY,
        connection_string=settings.DATABASE_URL,
        embedding_function=embeddings_model,
        use_jsonb=True
    )

    assert vs1 is mock_instance
    assert vs2 is mock_instance


# get retriever 테스트
def test_get_company_retriever_creation(mocker):
    mock_vectorstore_instance = MagicMock(spec=PGVector)
    mock_retriever_instance = MagicMock(spec=BaseRetriever)
    mock_vectorstore_instance.as_retriever.return_value = mock_retriever_instance

    mocker.patch('app.core.vector_db.get_company_vectorstore', return_value=mock_vectorstore_instance)

    retriever = get_company_retriever(top_k=5)

    mock_vectorstore_instance.as_retriever.assert_called_once_with(search_kwargs= {"k": 5})
    assert retriever is mock_retriever_instance

def test_get_news_retriever_creation(mocker):
    mock_vectorstore_instance = MagicMock(spec=PGVector)
    mock_retriever_instance = MagicMock(spec=BaseRetriever)
    mock_vectorstore_instance.as_retriever.return_value = mock_retriever_instance

    mocker.patch('app.core.vector_db.get_news_vectorstore', return_value=mock_vectorstore_instance)

    retriever = get_news_retriever(top_k=5)

    mock_vectorstore_instance.as_retriever.assert_called_once_with(search_kwargs= {"k": 5})
    assert retriever is mock_retriever_instance

def test_get_university_retriever_creation(mocker):
    mock_vectorstore_instance = MagicMock(spec=PGVector)
    mock_retriever_instance = MagicMock(spec=BaseRetriever)
    mock_vectorstore_instance.as_retriever.return_value = mock_retriever_instance

    mocker.patch('app.core.vector_db.get_university_vectorstore', return_value=mock_vectorstore_instance)

    retriever = get_university_retriever(top_k=5)

    mock_vectorstore_instance.as_retriever.assert_called_once_with(search_kwargs= {"k": 5})
    assert retriever is mock_retriever_instance


# retrieve_documents_from_sources 테스트
@pytest.fixture
def mock_retrievers(mocker):
    mock_company_ret_obj = AsyncMock(spec=BaseRetriever)
    mock_news_ret_obj = AsyncMock(spec=BaseRetriever)
    mock_univeristy_ret_obj = AsyncMock(spec=BaseRetriever)

    mocker.patch('app.core.vector_db.get_company_retriever',return_value=mock_company_ret_obj)
    mocker.patch('app.core.vector_db.get_news_retriever',return_value=mock_news_ret_obj)
    mocker.patch('app.core.vector_db.get_university_retriever',return_value=mock_univeristy_ret_obj)

    return mock_company_ret_obj, mock_news_ret_obj, mock_univeristy_ret_obj

@pytest.mark.asyncio
async def test_retrieve_documents_all_success(mock_retrievers: tuple):
    mock_company_ret, mock_news_ret, mock_university_ret = mock_retrievers
    doc_company1 = Document(page_content="CompanyDoc1")
    doc_news1 = Document(page_content="NewsDoc1")
    doc_univeristy1 = Document(page_content="UniversityDoc1")

    mock_company_ret.aget_relevant_documents.return_value=[doc_company1]
    mock_news_ret.aget_relevant_documents.return_value=[doc_news1]
    mock_university_ret.aget_relevant_documents.return_value=[doc_univeristy1]

    #함수 호출
    result_docs = await retrieve_documents_from_sources(
        query="일반 쿼리", 
        university_query="대학 쿼리", 
        top_k_per_source=1, 
        top_k_university=1
    )

    assert len(result_docs) == 3 # 모든 문서가 고유하다고 가정
    contents = [doc.page_content for doc in result_docs]
    assert "CompanyDoc1" in contents
    assert "NewsDoc1" in contents
    assert "UniversityDoc1" in contents
    
    mock_company_ret.aget_relevant_documents.assert_called_once_with("일반 쿼리")
    mock_news_ret.aget_relevant_documents.assert_called_once_with("일반 쿼리")
    mock_university_ret.aget_relevant_documents.assert_called_once_with("대학 쿼리")


@pytest.mark.asyncio
async def test_retrieve_documents_some_sources_empty(mock_retrievers: tuple):
    # 뉴스 및 대학 정보는 검색 결과 없음, 회사 정보만 있을 때
    mock_company_ret, mock_news_ret, mock_university_ret = mock_retrievers

    doc_company1 = Document(page_content="CompanyDoc1")
    mock_company_ret.aget_relevant_documents.return_value = [doc_company1]
    mock_news_ret.aget_relevant_documents.return_value = []
    mock_university_ret.aget_relevant_documents.return_value = []

    # When
    result_docs = await retrieve_documents_from_sources(query="쿼리", university_query="대학쿼리")

    # Then
    assert len(result_docs) == 1
    assert result_docs[0].page_content == "CompanyDoc1"

@pytest.mark.asyncio
async def test_retrieve_documents_no_university_query(mock_retrievers: tuple):
    # university_query가 None인 경우 대학 검색을 시도하지 않아야 할 때
    mock_company_ret, mock_news_ret, mock_university_ret = mock_retrievers

    doc_company1 = Document(page_content="CompanyDoc1")
    mock_company_ret.aget_relevant_documents.return_value = [doc_company1]
    mock_news_ret.aget_relevant_documents.return_value = []

    # When
    result_docs = await retrieve_documents_from_sources(query="쿼리", university_query=None, top_k_per_source=1, top_k_university=1)

    # Then
    assert len(result_docs) == 1
    assert result_docs[0].page_content == "CompanyDoc1"
    mock_university_ret.assert_not_called()

@pytest.mark.asyncio
async def test_retrieve_documents_duplicate_content(mock_retrievers: tuple):
    # Given
    mock_company_ret, mock_news_ret, _ = mock_retrievers # 대학은 사용 안한다고 가정하기
    doc_shared = Document(page_content="Shared Content")
    mock_company_ret.aget_relevant_documents.return_value = [doc_shared, Document(page_content="Company Unique")]
    mock_news_ret.aget_relevant_documents.return_value = [doc_shared, Document(page_content="News Unique")]

    # When
    results = await retrieve_documents_from_sources("query", None)

    # Then
    assert len(results) == 3 
    contents = {doc.page_content for doc in results}
    assert "Shared Content" in contents
    assert "Company Unique" in contents
    assert "News Unique" in contents
