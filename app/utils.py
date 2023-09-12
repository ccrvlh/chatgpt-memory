import json
import logging
import requests
import time

from random import random
from typing import Any, Optional
from typing import Tuple
from typing import Union
from transformers import GPT2TokenizerFast  # type: ignore[import]

from app.config import OPENAI_BACKOFF
from app.config import OPENAI_MAX_RETRIES
from app.config import OPENAI_TIMEOUT
from app.exceptions import OpenAIError
from app.exceptions import OpenAIRateLimitError


logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_name: str, use_tiktoken: bool) -> Any:
    """
    Load either the tokenizer from tiktoken (if the library is available) or
    fallback to the GPT2TokenizerFast from the transformers library.

    Args:
        tokenizer_name (str): The name of the tokenizer to load.
        use_tiktoken (bool): Use tiktoken tokenizer or not.

    Raises:
        ImportError: When `tiktoken` package is missing.
        To use tiktoken tokenizer install it as follows:
        `pip install tiktoken`

    Returns:
        tokenizer: Tokenizer of either GPT2 kind or tiktoken based.
    """
    tokenizer = None
    if use_tiktoken:
        try:
            import tiktoken  # pylint: disable=import-error

            logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
            tokenizer = tiktoken.get_encoding(tokenizer_name)
        except ImportError:
            raise ImportError(
                "The `tiktoken` package not found.",
                "To install it use the following:",
                "`pip install tiktoken`",
            )
    else:
        logger.warning(
            "OpenAI tiktoken module is not available for Python < 3.8, Linux ARM64 and "
            "AARCH64. Falling back to GPT2TokenizerFast."
        )

        logger.debug("Using GPT2TokenizerFast tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


def count_tokens(text: str, tokenizer: Any, use_tiktoken: bool) -> int:
    """
    Count the number of tokens in `text` based on the provided OpenAI `tokenizer`.

    Args:
        text (str):  A string to be tokenized.
        tokenizer (Any): An OpenAI tokenizer.
        use_tiktoken (bool): Use tiktoken tokenizer or not.

    Returns:
        int: Number of tokens in the text.
    """
    tokens = len(tokenizer.tokenize(text))
    if use_tiktoken:
        tokens = len(tokenizer.encode(text))
    return tokens


def get_prompt(message: str, history: str) -> str:
    """
    Generates the prompt based on the current history and message.

    Args:
        message (str): Current message from user.
        history (str): Retrieved history for the current message.
        History follows the following format for example:
        ```
        Human: hello
        Assistant: hello, how are you?
        Human: good, you?
        Assistant: I am doing good as well. How may I help you?
        ```
    Returns:
        prompt: Curated prompt for the ChatGPT API based on current params.
    """
    prompt = f"""Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    {history}
    Human: {message}
    Assistant:"""

    return prompt


def retry(
    backoff_in_seconds: float = OPENAI_BACKOFF,
    max_retries: int = OPENAI_MAX_RETRIES,
    errors: tuple = (OpenAIRateLimitError, OpenAIError),
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        backoff_in_seconds: The initial backoff in seconds.
        max_retries: The maximum number of retries.
        errors: The errors to catch retry on.
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0

            # Loop until a successful response or max_retries is hit or an
            # exception is raised
            while True:
                try:
                    return function(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    sleep_time = backoff_in_seconds * 2**num_retries + random()

                    # Sleep for the delay
                    logger.warning(
                        "%s - %s, retry %s in %s seconds...",
                        e.__class__.__name__,
                        e,
                        function.__name__,
                        "{0:.2f}".format(sleep_time),
                    )
                    time.sleep(sleep_time)

                    # Increment retries
                    num_retries += 1

        return wrapper

    return decorator


@retry(backoff_in_seconds=15)
def openai_request(
    url: str,
    api_key: str,
    payload: dict,
    headers: Optional[dict] = None,
    timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT,
) -> dict:
    """
    Make a request to the OpenAI API given a `url`, `headers`, `payload`, and
    `timeout`. If request is unsucessful and `status_code = 429` raise rate limiting error else the OpenAIError.

    Args:
        url (str): The URL of the OpenAI API.
        headers (Dict): Dictionary of HTTP Headers to send with the :class:`Request`.
        payload (Dict): The payload to send with the request.
        timeout (Union[float, Tuple[float, float]], optional): The timeout length of the request. The default is 30s.
        Defaults to OPENAI_TIMEOUT.

    Raises:
        openai_error: If the request fails.

    Returns:
        Dict: OpenAI Embedding API response.
    """
    if not headers:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.request(
        "POST", url, headers=headers, data=json.dumps(payload), timeout=timeout
    )
    res = json.loads(response.text)

    if response.status_code != 200:
        openai_error: OpenAIError
        if response.status_code == 429:
            openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
        else:
            openai_error = OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error

    return res
