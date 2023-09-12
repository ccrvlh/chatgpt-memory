import pytest

from app.datastore.config import RedisDataStoreConfig
from app.redis import RedisDataStore
from app.environment import OPENAI_API_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from app.llm_client.openai.embedding.config import EmbeddingConfig
from app.llm_client.openai.embedding.embedding_client import EmbeddingClient


@pytest.fixture(scope="session")
def openai_embedding_client():
    embedding_config = EmbeddingConfig(api_key=OPENAI_API_KEY)
    return EmbeddingClient(config=embedding_config)


@pytest.fixture(scope="session")
def redis_datastore():
    redis_datastore_config = RedisDataStoreConfig(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
    )
    redis_datastore = RedisDataStore(config=redis_datastore_config, do_flush_data=True)

    return redis_datastore
