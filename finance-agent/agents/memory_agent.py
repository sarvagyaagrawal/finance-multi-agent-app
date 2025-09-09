from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import HuggingFaceEndpoint
from langchain_experimental.sql import SQLDatabaseChain
from typing import Dict, Any, Optional

import streamlit as st


class MemoryAgent:
    def __init__(self, llm=None, hf_token: Optional[str] = None):
        # Store Hugging Face token
        self.hf_token = hf_token

        # Setup message history in Streamlit session
        self.msgs = StreamlitChatMessageHistory(key="langchain_messages")
        if len(self.msgs.messages) == 0:
            self.msgs.add_ai_message("How can I help you?")

        # Define prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI chatbot having a helpful, friendly conversation with a human."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # HuggingFace LLM (replace repo_id with your best conversational model)
        # self.llm = HuggingFaceEndpoint(
        #     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        #     huggingfacehub_api_token=self.hf_token,
        #     temperature=0.7,
        #     max_new_tokens=512,
        # )
        self.llm = llm or HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=self.hf_token,
            temperature=0.7,
            max_new_tokens=512,
        )

        # Combine prompt + LLM
        self.chain = self.prompt | self.llm

        # Add memory wrapper
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.msgs,
            input_messages_key="question",
            history_messages_key="history",
        )

    def run(self, query: str, context=None) -> str:
        """Run the memory agent with conversational context."""
        config = {"configurable": {"session_id": "user"}}
        response = self.chain_with_history.invoke({"question": query}, config)
        return response if isinstance(response, str) else response.content
