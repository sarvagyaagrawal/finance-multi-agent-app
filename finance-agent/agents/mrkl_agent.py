# agents/mrkl_agent.py
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine
from typing import Dict, Any, Optional
import os
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import LLMMathChain
from langchain_community.utilities import SearchApiAPIWrapper, SQLDatabase
from langchain import hub
from langchain_community.llms import HuggingFaceEndpoint
from langchain_experimental.sql import SQLDatabaseChain
import streamlit as st
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler


DB_PATH = (Path(__file__).parent.parent / "Chinook.db").absolute()

print("DB_PATH", DB_PATH)
class MrklAgent:
    def __init__(self, llm=None,hf_token: Optional[str] = None):
        # Hugging Face LLM (replaces OpenAI)
        self.llm = llm or HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0,
            huggingfacehub_api_token=hf_token,
            max_new_tokens=256,
        )

        # Tools setup
        os.environ["SEARCHAPI_API_KEY"] = "cDchfbAPk25MBnu1GVxdXqKC"
        search = SearchApiAPIWrapper()
        llm_math_chain = LLMMathChain.from_llm(self.llm)

        # Make the DB connection read-only
        creator = lambda: sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        db = SQLDatabase(create_engine("sqlite:///", creator=creator))
        db_chain = SQLDatabaseChain.from_llm(self.llm, db)

        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful for answering current events questions. Ask targeted questions.",
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="Useful for math-related queries.",
            ),
            Tool(
                name=" DB",
                func=db_chain.run,
                description="Useful for answering questions about database.",
            ),
        ]

        react_agent = create_react_agent(self.llm, tools, hub.pull("hwchase17/react"))
        self.agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def run(self, query: str, context=None) -> str:
        """Run MRKL agent with a user query."""
        try:
             # Create a container in Streamlit for live messages
            container = st.empty()
            callback = StreamlitCallbackHandler(container)
            cfg = RunnableConfig()
            cfg["callbacks"] = [callback]

            # Pass the callback to invoke
            result = self.agent_executor.invoke({"input": query}, cfg)

            # If result has "output", return it; else convert to string
            if isinstance(result, dict) and "output" in result:
                container.write(result["output"])
                return result["output"]
            elif hasattr(result, "content"):
                return result.content
                
            else:
                return str(result)
           
        except Exception as e:
            return f"⚠️ MRKL Agent failed: {e}"
        
        
        
