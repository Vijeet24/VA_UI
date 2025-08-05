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
        return f"Error loading CSV: {e}"

def main():
    load_dotenv()
    st.set_page_config(page_title="Voice Assistant | I-CPS Lab", layout="wide")
    st.title("🎙️ Voice Assistant Interface - I-CPS Lab, Polytechnique Montréal")

    # Define CSV path
    local_csv_path = r"C:/Users/ACER/langchain-ask-csv/data.csv"  # Use raw string to avoid escape errors

    # Initialize memory and agent once
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if 'agent' not in st.session_state and os.path.exists(local_csv_path):
        st.session_state.agent = create_csv_agent(
            llm=OpenAI(temperature=0),
            path=local_csv_path,
            verbose=False,
            memory=st.session_state.memory,
            allow_dangerous_code=True
        )

    # Layout: Left = Live Data View, Right = Chat Interface
    col1, col2 = st.columns([1, 2])

    # Left: Live Data Viewer (auto-refresh CSV every second)
    with col1:
        st.subheader("📊 Live CSV Data (Auto-updating)")
        data_placeholder = st.empty()

        def update_csv_data():
            while True:
                if os.path.exists(local_csv_path):
                    df = load_csv_data(local_csv_path)
                    data_placeholder.dataframe(df)
                time.sleep(1)

        # Start the background thread only once
        if 'thread_started' not in st.session_state:
            threading.Thread(target=update_csv_data, daemon=True).start()
            st.session_state.thread_started = True

    # Right: Continuous Chat
    with col2:
        st.subheader("💬 Continuous Chat with CSV")

        user_input = st.text_input("Ask something about the CSV data:", key="chat_input")

        if user_input:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(user_input)
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Assistant", response))
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display past conversation
        if "chat_history" in st.session_state:
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"**🧑 You:** {message}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message}")

if __name__ == "__main__":
    main()

