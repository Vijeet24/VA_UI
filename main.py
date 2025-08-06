from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os

# To address the LangChainDeprecationWarning
from langchain_openai import OpenAI as LangchainOpenAI

def load_csv_data(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def main():
    load_dotenv()
    st.set_page_config(page_title="Voice Assistant | I-CPS Lab", layout="wide")
    st.title("🎙️ Voice Assistant Interface - I-CPS Lab, Polytechnique Montréal")

    # Define CSV path
    local_csv_path = "data.csv"

    # ---
    # NEW: Initialize the agent AND the chat history in the same place.
    # ---

    if 'agent' not in st.session_state:
        if os.path.exists(local_csv_path):
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            # Using the new OpenAI class to address the deprecation warning
            st.session_state.agent = create_csv_agent(
                llm=LangchainOpenAI(temperature=0),
                path=local_csv_path,
                verbose=False,
                allow_dangerous_code=True
            )
            # The memory parameter is deprecated in create_csv_agent, so we will manage it manually
            # in the chat logic
            st.session_state.chat_history = []
        else:
            st.error(f"Could not find the CSV file at: {local_csv_path}. Please check the path.")
            st.session_state.agent = None

    # Layout: Left = Live Data View, Right = Chat Interface
    col1, col2 = st.columns([1, 2])

    # Left: Live Data Viewer
    with col1:
        st.subheader("📊 Live CSV Data")

        # Load the data directly in the main script thread
        df = load_csv_data(local_csv_path)
        if df is not None:
            st.dataframe(df)

        # You can add a button to manually refresh the data
        if st.button("Refresh Data"):
            st.rerun()

    # Right: Continuous Chat
    with col2:
        st.subheader("💬 Continuous Chat with CSV")

        # Only display the chat interface if the agent was successfully initialized
        if st.session_state.agent is not None:
            # Use the new st.chat_input widget
            user_input = st.chat_input("Ask something about the CSV data:")

            # Display past conversation
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    with st.chat_message("user"):
                        st.markdown(message)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message)

            if user_input:
                with st.spinner("Thinking..."):
                    try:
                        # Append the user message to the history first
                        st.session_state.chat_history.append(("You", user_input))
                        # Use the agent's memory (which is now part of the LLM call)
                        response = st.session_state.agent.run(user_input, memory=st.session_state.memory)
                        st.session_state.chat_history.append(("Assistant", response))
                        # Rerun the app to show the new messages
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("Chat functionality is disabled because the CSV file was not found.")

if __name__ == "__main__":
    main()

