import streamlit as st
from controller.controller_agent import ControllerAgent

st.title("📊 Finance Multi-Agent App")

# Step 1: Ask user for Hugging Face token
hf_token = st.text_input("🔑 Enter your Hugging Face token:", type="password")

if hf_token:
    # Step 2: Initialize controller with token
    controller = ControllerAgent(hf_token=hf_token)

    # query = st.text_input("Ask me anything:")
    # if query:
    #     response = controller.route(query)
    #     st.write(response)

    if query := st.chat_input("Ask me anything:"):
        result = controller.route(query)

        st.subheader("🧠 Controller Agent Thinking")
        st.code(result["thinking"], language="text")

        st.subheader("📌 Routing Decision")
        st.write(result["decision"])

        st.subheader("🤖 Final Answer")
        st.write(result["answer"])
else:
    st.warning("Please enter your Hugging Face token to continue.")
