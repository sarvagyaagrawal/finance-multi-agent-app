from typing import Dict, Any
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from agents.csv_agent import CsvAgent
from agents.mrkl_agent import MrklAgent
from agents.memory_agent import MemoryAgent

class ControllerAgent:
    def __init__(self, hf_token: str):
        # Hugging Face LLM
        llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1-0528",
            temperature=0,
            max_new_tokens=256,
            huggingfacehub_api_token=hf_token,
            provider="auto",  # passed from UI
            task="conversational",
        )
        self.llm = ChatHuggingFace(llm=llm)

        template = """
        You are a controller agent. Route the user query to the correct specialist agent.

        Agents:
        - csv : Handles CSV, Pandas, data analysis tasks
        - mrkl : Handles reasoning + external tools (math, search, APIs)
        - memory : Handles conversational Q&A, context, and follow-up chats

        User query: {query}

        Respond ONLY with one word: 'csv', 'mrkl', or 'memory'.
        """
       
        self.prompt = PromptTemplate.from_template(template)

        # self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.llm_chain = self.prompt | self.llm

        # Specialist agents
        self.csv_agent = CsvAgent(self.llm)
        self.mrkl_agent = MrklAgent(self.llm)
        self.memory_agent = MemoryAgent(self.llm)
    
    def extract_controller_decision(self, response: str) -> str:
        """
        Extract the final agent decision from the Controller response.
        Assumes model outputs something like:
        <think>...reasoning...</think>\n<agent_name>
        """
        # Try to find an agent name after the last </think>
        match = re.search(r"</think>\s*(\w+)", response, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        # fallback: use last line as agent
        return response.strip().split("\n")[-1].lower()

    def route(self, query: str, context: Dict[str, Any] = None) -> str:

        decision = self.llm_chain.invoke({"query":query})
        raw_output = decision.content
        print(f"Raw controller decision: {decision} with type {type(decision)}")
        agent_name = self.extract_controller_decision(decision.content)
        print(f"Controller decision: {agent_name}")
        
        
        if "csv" in agent_name:
            answer = self.csv_agent.run(query)
        elif "mrkl" in agent_name:
            answer = self.mrkl_agent.run(query)
        elif "memory" in agent_name:
            answer = self.memory_agent.run(query)
        else:
            answer = f"❓ Sorry, I could not route your query. Decision was: {agent_name}"

        return {
            "thinking": raw_output,   # full response including <think>...</think>
            "decision": agent_name,   # extracted routing word
            "answer": answer          # final answer from sub-agent
        }

