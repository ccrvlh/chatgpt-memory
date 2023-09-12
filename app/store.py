import numpy as np

from typing import Any

from app.redis import RedisDataStore
from app.embeddings import EmbeddingClient


class MemoryManager:
    """
    Manages the memory of conversations.

    Attributes:
        datastore (DataStore): Datastore to use for storing and retrieving memories.
        embed_client (EmbeddingClient): Embedding client to call for embedding conversations.
        conversations_ids (list[str]): list of conversation IDs to memories to be managed.
    """

    def __init__(
        self, datastore: RedisDataStore, embed_client: EmbeddingClient, topk: int = 100
    ) -> None:
        """
        Initializes the memory manager.

        Args:
            datastore (DataStore): Datastore to be used. Assumed to be connected.
            embed_client (EmbeddingClient): Embedding client to be used.
            topk (int): Number of past message to be retrieved as context for current message.
        """
        self.datastore = datastore
        self.embed_client = embed_client
        self.topk = topk
        self.conversations_ids: list[str] = [_id for _id in datastore.get_all_conversation_ids()]

    def add_conversation(self, conversation_id: str) -> None:
        """
        Adds a conversation to the memory manager to be stored and manage.

        Args:
            conversation (Memory): Conversation to be added.
        """
        if conversation_id not in self.conversations_ids:
            self.conversations_ids.append(conversation_id)

    def remove_conversation(self, conversation_id: str) -> None:
        """
        Removes a conversation from the memory manager.

        Args:
            conversation (Memory): Conversation to be removed containing `conversation_id`.
        """
        if conversation_id not in self.conversations_ids:
            return

        conversation_idx = self.conversations_ids.index(conversation_id)
        if conversation_idx >= 0:
            del self.conversations_ids[conversation_idx]
            self.datastore.delete_documents(conversation_id)

    def get_messages(self, conversation_id: str, query: str) -> list[Any]:
        """
        Gets the messages of a conversation using the query message.

        Args:
            conversation_id (str): ID of the conversation to get the messages of.
            query (str): Current user message you want to pull history for to use in the prompt.
            topk (int): Number of messages to be returned. Defaults to 5.

        Returns:
            list[Any]: list of messages of the conversation.
        """
        if conversation_id not in self.conversations_ids:
            raise ValueError(
                f"Conversation id: {conversation_id} is not present in past conversations."
            )

        embed_array = self.embed_client.embed_queries([query])
        query_vector = embed_array[0].astype(np.float32).tobytes()
        messages = self.datastore.search_documents(
            query_vector=query_vector, conversation_id=conversation_id, topk=self.topk
        )
        return messages

    def add_message(self, conversation_id: str, human: str, assistant: str) -> None:
        """
        Adds a message to a conversation.

        Args:
            conversation_id (str): ID of the conversation to add the message to.
            human (str): User message.
            assistant (str): Assistant message.
        """
        document: dict[str, str] = {
            "text": f"Human: {human}\nAssistant: {assistant}",
            "conversation_id": conversation_id,
        }
        embedded_docs = self.embed_client.embed_documents(docs=[document])
        document["embedding"] = embedded_docs[0].astype(np.float32).tobytes()
        self.datastore.index_documents(documents=[document])
        self.add_conversation(conversation_id)

    def clear(self) -> None:
        """
        Clears the memory manager.
        """
        self.datastore.flush_all_documents()
        self.conversations_ids = []

    def __del__(self) -> None:
        """Clear the memory manager when manager is deleted."""
        self.clear()
