import logging
import numpy as np

from tqdm import tqdm
from typing import Any
from typing import Union

from app.config import MAX_ALLOWED_SEQ_LEN_001
from app.config import MAX_ALLOWED_SEQ_LEN_002
from app.schemas import EmbeddingModels
from app.schemas import EmbeddingConfig
from app.utils import count_tokens
from app.utils import load_tokenizer
from app.utils import openai_request


logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self._api_key = config.api_key
        self._time_out = config.time_out
        self.config = config
        model_class: str = EmbeddingModels(self.config.model).name

        tokenizer = self._setup_encoding_models(
            model_class,
            self.config.model,
            self.config.max_seq_len,
        )
        self._tokenizer = load_tokenizer(
            tokenizer_name=tokenizer,
            use_tiktoken=self.config.use_tiktoken,
        )

    @property
    def api_key(self):
        return self._api_key

    @property
    def time_out(self):
        return self._time_out

    def _setup_encoding_models(self, model_class: str, model_name: str, max_seq_len: int):
        """
        Setup the encoding models for the retriever.

        Raises:
            ImportError: When `tiktoken` package is missing.
            To use tiktoken tokenizer install it as follows:
            `pip install tiktoken`
        """

        tokenizer_name = "gpt2"
        # new generation of embedding models (December 2022), specify the full name
        if model_name.endswith("-002"):
            self.query_encoder_model = model_name
            self.doc_encoder_model = model_name
            self.max_seq_len = min(MAX_ALLOWED_SEQ_LEN_002, max_seq_len)
            if self.config.use_tiktoken:
                try:
                    from tiktoken.model import MODEL_TO_ENCODING  # type: ignore[attr-defined]

                    tokenizer_name = MODEL_TO_ENCODING.get(model_name, "cl100k_base")
                except ImportError:
                    raise ImportError(
                        "The `tiktoken` package not found.",
                        "To install it use the following:",
                        "`pip install tiktoken`",
                    )
        else:
            self.query_encoder_model = f"text-search-{model_class}-query-001"
            self.doc_encoder_model = f"text-search-{model_class}-doc-001"
            self.max_seq_len = min(MAX_ALLOWED_SEQ_LEN_001, max_seq_len)

        return tokenizer_name

    def _ensure_text_limit(self, text: str) -> str:
        """
         Ensure that length of the text is within the maximum length of the model.
        OpenAI v1 embedding models have a limit of 2046 tokens, and v2 models have
        a limit of 8191 tokens.

        Args:
            text (str):  Text to be checked if it exceeds the max token limit

        Returns:
            text (str): Trimmed text if exceeds the max token limit
        """
        n_tokens = count_tokens(text, self._tokenizer, self.config.use_tiktoken)
        if n_tokens <= self.max_seq_len:
            return text

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens to fit"
            "within the max token limit.",
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_tokens,
            self.max_seq_len,
        )

        if self.config.use_tiktoken:
            tokenized_payload = self._tokenizer.encode(text)
            decoded_string = self._tokenizer.decode(tokenized_payload[: self.max_seq_len])
        else:
            tokenized_payload = self._tokenizer.tokenize(text)
            decoded_string = self._tokenizer.convert_tokens_to_string(
                tokenized_payload[: self.max_seq_len]
            )

        return decoded_string

    def embed(self, model: str, text: list[str]) -> np.ndarray:
        """
        Embeds the batch of texts using the specified LLM.

        Args:
            model (str): LLM model name for embeddings.
            text (list[str]): list of documents to be embedded.

        Raises:
            ValueError: When the OpenAI API key is missing.

        Returns:
            np.ndarray: embeddings for the input documents.
        """
        if self.api_key is None:
            raise ValueError(
                "OpenAI API key is not set. You can set it via the "
                "`api_key` parameter of the `LLMClient`."
            )

        generated_embeddings: list[Any] = []
        payload: dict[str, Union[list[str], str]] = {"model": model, "input": text}
        res = openai_request(
            url=self.config.endpoint,
            api_key=self.api_key,
            payload=payload,
            timeout=self.time_out,
        )
        data = res.get("data", [])
        unordered_embeddings = [(ans["index"], ans["embedding"]) for ans in data]
        ordered_embeddings = sorted(unordered_embeddings, key=lambda x: x[0])
        generated_embeddings = [emb[1] for emb in ordered_embeddings]
        return np.array(generated_embeddings)

    def embed_batch(self, model: str, text: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(
            range(0, len(text), self.config.batch_size),
            disable=not self.config.progress_bar,
            desc="Calculating embeddings",
        ):
            batch = text[i : i + self.config.batch_size]
            batch_limited = [self._ensure_text_limit(content) for content in batch]
            generated_embeddings = self.embed(model, batch_limited)
            all_embeddings.append(generated_embeddings)

        return np.concatenate(all_embeddings)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self.embed_batch(self.query_encoder_model, queries)

    def embed_documents(self, docs: list[dict]) -> np.ndarray:
        return self.embed_batch(self.doc_encoder_model, [d["text"] for d in docs])
