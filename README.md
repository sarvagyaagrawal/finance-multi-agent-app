# Finance Multi-Agent App

A Streamlit-based multi-agent system for finance and data tasks, built using **LangChain**, **Hugging Face models**, and **Poetry** for dependency management.  
No OpenAI keys required — everything runs with Hugging Face inference endpoints.

---

## 🚀 Features

- **Controller Agent**: Routes queries to the right specialist agent (CSV, MRKL, or Memory).
- **CSV Agent**: Analyzes CSV files with Pandas.
- **MRKL Agent**: Handles reasoning, math, search, and SQL queries (Chinook database included).
- **Memory Agent**: Supports conversational Q&A and context.
- **Error Handling**: Graceful fallbacks with clear error messages.
- **UI**: Streamlit frontend with simple query + result display.

---

## 🛠️ Tech Stack

- [Python 3.10+](https://www.python.org/downloads/)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Poetry](https://python-poetry.org/)
- Hugging Face Hub (DeepSeek, Mistral, or any supported LLMs)
- DuckDuckGo Search API Wrapper
- SQLite (Chinook sample database)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/finance-agent-app.git
cd finance-agent-app
```

Install dependencies with **Poetry**:

```bash
poetry install
```

Activate the environment:

```bash
poetry shell
```

---

## 🔑 Hugging Face Token Setup

You’ll need a [Hugging Face access token](https://huggingface.co/settings/tokens).  
We recommend using a **fine-grained token** with `inference` permissions only.

Set it in your environment:

**Linux / macOS:**
```bash
export HF_TOKEN="your_token_here"
```

**Windows (PowerShell):**
```powershell
setx HF_TOKEN "your_token_here"
```

---

## ▶️ Run the App

Start Streamlit:

```bash
streamlit run app.py
```

The app will reload automatically in **debug mode** when you update the code.

---

## 💡 Example Queries

- `"Upload a CSV and find the average revenue"`
- `"What is 145 * 98?"`
- `"CEO of Tesla"`
- `"Tell me what I just asked before"`

The controller agent decides:
- **CSV Agent** → for CSV/data analysis tasks  
- **MRKL Agent** → for reasoning, search, SQL, or math  
- **Memory Agent** → for conversational context  

---

## 📂 Project Structure

```
finance-agent-app/
│
├── app.py                   # Streamlit entrypoint
├── controller/
│   └── controller_agent.py  # Controller Agent
├── agents/
│   ├── csv_agent.py
│   ├── mrkl_agent.py
│   └── memory_agent.py
├── Chinook.db               # Sample SQLite database
├── pyproject.toml           # Poetry dependencies
└── README.md
```

---

## ⚠️ Error Handling

- All agents include try/except blocks with clear error messages (`⚠️ MRKL Agent failed: ...`).
- Controller gracefully handles unknown routing decisions.
- SearchAPI search fallback ensures queries don’t crash the app.

---

## 📝 License

MIT License. Feel free to use, modify, and share.
