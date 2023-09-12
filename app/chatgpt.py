import logging
import uuid

from typing import Optional
from langchain import LLMChain
from langchain import OpenAI
from langchain import PromptTemplate

from app.store import MemoryManager
from app.utils import get_prompt
from app.schemas import ChatGPTConfig
from app.schemas import ChatGPTResponse


logger = logging.getLogger(__name__)


class ChatGPTClient:
    """
    ChatGPT client allows to interact with the ChatGPT model alonside having infinite contextual and adaptive memory.

    """

    def __init__(self, config: ChatGPTConfig, store: MemoryManager):
        self._api_key = config.api_key
        self._time_out = config.time_out
        prompt = PromptTemplate(input_variables=["prompt"], template="{prompt}")
        openai_client = OpenAI(
            temperature=config.temperature,
            openai_api_key=self.api_key,
            model_name=config.model_name,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
            client=None,
        )
        self.chain = LLMChain(
            llm=openai_client,
            prompt=prompt,
            verbose=config.verbose,
        )
        self.store = store

    @property
    def api_key(self):
        return self._api_key

    @property
    def time_out(self):
        return self._time_out

    def converse(self, message: str, conversation_id: Optional[str] = None) -> ChatGPTResponse:
        """
        Allows user to chat with user by leveraging the infinite contextual memor for fetching and
        adding historical messages to the prompt to the ChatGPT model.

        Args:
            message (str): Message by the human user.
            conversation_id (str, optional): Id of the conversation, if session already exists. Defaults to None.

        Returns:
            ChatGPTResponse: Response includes answer from th ChatGPT, conversation_id, and human message.
        """
        if not conversation_id:
            conversation_id = uuid.uuid4().hex

        history = ""
        try:
            past_messages = self.store.get_messages(conversation_id=conversation_id, query=message)
            history = "\n".join(
                [
                    past_message.text
                    for past_message in past_messages
                    if getattr(past_message, "text")
                ]
            )
        except ValueError as history_not_found_error:
            logger.warning(
                f"No previous chat history found for conversation_id: {conversation_id}.\nDetails: {history_not_found_error}"
            )
        prompt = get_prompt(message=message, history=history)
        chat_gpt_answer = self.chain.predict(prompt=prompt)

        if len(message.strip()) and len(chat_gpt_answer.strip()):
            self.store.add_message(
                conversation_id=conversation_id, human=message, assistant=chat_gpt_answer
            )

        response = ChatGPTResponse(
            message=message, chat_gpt_answer=chat_gpt_answer, conversation_id=conversation_id
        )
        return response
