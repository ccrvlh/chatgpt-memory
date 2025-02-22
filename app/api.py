from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from app.schemas import ChatGPTConfig
from app.schemas import RedisDataStoreConfig
from app.redis import RedisDataStore
from app.config import OPENAI_API_KEY
from app.config import REDIS_HOST
from app.config import REDIS_PASSWORD
from app.config import REDIS_PORT
from app.chatgpt import ChatGPTClient
from app.chatgpt import ChatGPTResponse
from app.embeddings import EmbeddingClient
from app.embeddings import EmbeddingConfig
from app.store import MemoryManager


if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

if not REDIS_HOST:
    raise ValueError("REDIS_HOST is not set")

if not REDIS_PORT:
    raise ValueError("REDIS_PORT is not set")

if not REDIS_PASSWORD:
    raise ValueError("REDIS_PASSWORD is not set")


# Instantiate an EmbeddingConfig object with the OpenAI API key
# Instantiate an EmbeddingClient object with the EmbeddingConfig object
embedding_config = EmbeddingConfig(api_key=OPENAI_API_KEY)
embed_client = EmbeddingClient(config=embedding_config)

# Instantiate a RedisDataStoreConfig object with the Redis connection details
redis_datastore_config = RedisDataStoreConfig(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
)

# Instantiate a RedisDataStore object with the RedisDataStoreConfig object
redis_datastore = RedisDataStore(config=redis_datastore_config)

# Instantiate a MemoryManager object with the RedisDataStore object and EmbeddingClient object
memory_manager = MemoryManager(datastore=redis_datastore, embed_client=embed_client, topk=1)

# Instantiate a ChatGPTConfig object with the OpenAI API key and verbose set to True
chat_gpt_config = ChatGPTConfig(api_key=OPENAI_API_KEY, verbose=False)

# Instantiate a ChatGPTClient object with the ChatGPTConfig object and MemoryManager object
chat_gpt_client = ChatGPTClient(config=chat_gpt_config, store=memory_manager)


class MessagePayload(BaseModel):
    conversation_id: Optional[str]
    message: str


app = FastAPI()


@app.post("/converse")
async def converse(message_payload: MessagePayload) -> ChatGPTResponse:
    response = chat_gpt_client.converse(**message_payload.dict())
    return response
