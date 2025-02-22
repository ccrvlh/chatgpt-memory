from enum import Enum
from pydantic import BaseModel


class EmbeddingModels(Enum):
    Ada = "*-ada-*-001"
    Babbage = "*-babbage-*-001"
    Curie = "*-curie-*-001"
    DaVinci = "*-davinci-*-001"


class RedisIndexType(Enum):
    HNSW = "HNSW"
    Flat = "FLAT"


class LLMClientConfig(BaseModel):
    api_key: str
    time_out: float = 30


class ChatGPTConfig(BaseModel):
    api_key: str
    time_out: float = 30
    temperature: float = 0
    model_name: str = "gpt-3.5-turbo"
    max_retries: int = 6
    max_tokens: int = 256
    verbose: bool = False


class EmbeddingConfig(BaseModel):
    api_key: str
    time_out: float = 30
    endpoint: str = "https://api.openai.com/v1/embeddings"
    batch_size: int = 64
    progress_bar: bool = False
    model: str = EmbeddingModels.Ada.value
    max_seq_len: int = 8191
    use_tiktoken: bool = False


class DataStoreConfig(BaseModel):
    host: str
    port: int
    password: str
    index_type: str = RedisIndexType.HNSW.value
    vector_field_name: str = "embedding"
    vector_dimensions: int = 1024
    distance_metric: str = "L2"
    number_of_vectors: int = 686
    M: int = 40
    EF: int = 200


class ChatGPTResponse(BaseModel):
    conversation_id: str
    message: str
    chat_gpt_answer: str


class RedisDataStoreConfig(DataStoreConfig):
    pass
