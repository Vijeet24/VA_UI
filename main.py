from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import time
import threading

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
    # We will also add an `else` block to show an error if the CSV is missing.
    # ---

    if 'agent' not in st.session_state:
        if os.path.exists(local_csv_path):
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.agent = create_csv_agent(
                llm=OpenAI(temperature=0),
                path=local_csv_path,
                verbose=False,
                memory=st.session_state.memory,
                allow_dangerous_code=True
            )
            # Initialize chat history here, as it's part of the agent's state
            st.session_state.chat_history = []
        else:
            st.error(f"Could not find the CSV file at: {local_csv_path}. Please check the path.")
            st.session_state.agent = None # Set agent to None to prevent subsequent errors

    # Layout: Left = Live Data View, Right = Chat Interface
    col1, col2 = st.columns([1, 2])

    # Left: Live Data Viewer
    with col1:
        st.subheader("📊 Live CSV Data (Auto-updating)")
        data_placeholder = st.empty()

        def update_csv_data():
            while True:
                df = load_csv_data(local_csv_path)
                if df is not None:
                    data_placeholder.dataframe(df)
                time.sleep(1)

        if 'thread_started' not in st.session_state:
            if st.session_state.agent is not None:
                threading.Thread(target=update_csv_data, daemon=True).start()
                st.session_state.thread_started = True

    # Right: Continuous Chat
    with col2:
        st.subheader("💬 Continuous Chat with CSV")

        # Only display the chat interface if the agent was successfully initialized
        if st.session_state.agent is not None:
            user_input = st.text_input("Ask something about the CSV data:", key="chat_input")
            
            # Display past conversation
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"**🧑 You:** {message}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message}")

            if user_input:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.agent.run(user_input)
                        st.session_state.chat_history.append(("You", user_input))
                        st.session_state.chat_history.append(("Assistant", response))
                        # Rerun the app to show the new messages
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("Chat functionality is disabled because the CSV file was not found.")

if __name__ == "__main__":
    main()

