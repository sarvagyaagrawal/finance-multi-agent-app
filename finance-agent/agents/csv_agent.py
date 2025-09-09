from typing import Dict, Any, Optional
import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint


class CsvAgent:
    def __init__(self, llm=None, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.df = None
        self.llm = llm or HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=self.hf_token,
            temperature=0.7,
            max_new_tokens=512,
        )
        self.agent = None
        

    def load_file(self, file_path: str):
        """Load CSV/Excel file into pandas DataFrame."""
        ext = file_path.split(".")[-1].lower()
        file_formats = {
            "csv": pd.read_csv,
            "xls": pd.read_excel,
            "xlsx": pd.read_excel,
            "xlsm": pd.read_excel,
            "xlsb": pd.read_excel,
        }

        if ext in file_formats:
            self.df = file_formats[ext](file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def setup_agent(self):
        """Setup LangChain Pandas agent after DataFrame is loaded."""
        if self.df is None:
            raise ValueError("No DataFrame loaded. Call `load_file()` first.")

      

        # self.llm = ChatOpenAI(
        #     temperature=0,
        #     model="gpt-3.5-turbo-0613",
        #     openai_api_key=self.api_key,
        # )

        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

    def run(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Answer questions about the loaded DataFrame.
        `context` can include {"file_path": "..."} to dynamically load data.
        """
        if context and "file_path" in context and self.df is None:
            self.load_file(context["file_path"])
            self.setup_agent()

        if not self.agent:
            raise ValueError("CSV agent not initialized. Call `setup_agent()` first.")

        try:
            return self.agent.run(query)
        except Exception as e:
            return f"⚠️ Error while processing query in CSV agent: {e}"
