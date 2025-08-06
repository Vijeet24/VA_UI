from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os

def load_csv_data(path):
    """
    Loads data from a local CSV file.
    Uses st.cache_data to cache the data, which prevents re-loading on every rerun.
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def initialize_agent_and_history(csv_path):
    """
    Initializes the LangChain agent and chat history in Streamlit's session state.
    """
    if os.path.exists(csv_path):
        # We will manage the memory manually as create_csv_agent does not
        # support it directly in this version.
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.agent = create_csv_agent(
            llm=LangchainOpenAI(temperature=0),
            path=csv_path,
            verbose=False,
            allow_dangerous_code=True
        )
        st.session_state.chat_history = []
        return True
    else:
        st.error(f"Could not find the CSV file at: {csv_path}. Please check the path.")
        st.session_state.agent = None
        return False

def main():
    """
    The main function for the Streamlit application.
    """
    load_dotenv()
    st.set_page_config(page_title="Voice Assistant | I-CPS Lab", layout="wide")
    st.title("🎙️ Voice Assistant Interface - I-CPS Lab, Polytechnique Montréal")

    local_csv_path = "data.csv"

    # Initialize the agent and history only once
    if 'agent' not in st.session_state:
        initialize_agent_and_history(local_csv_path)

    # Layout: Left = Live Data View, Right = Chat Interface
    col1, col2 = st.columns([1, 2])

    # Left: Live Data Viewer
    with col1:
        st.subheader("📊 Live CSV Data")
        
        # Load the data and display it in a dataframe
        df = load_csv_data(local_csv_path)
        if df is not None:
            st.dataframe(df)

        # A button to manually refresh the data
        if st.button("Refresh Data"):
            st.rerun()

    # Right: Continuous Chat
    with col2:
        st.subheader("💬 Continuous Chat with CSV")

        # Only display the chat interface if the agent was successfully initialized
        if st.session_state.agent is not None:
            # Display past conversation
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    with st.chat_message("user"):
                        st.markdown(message)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message)

            # Use st.chat_input for a better user experience
            user_input = st.chat_input("Ask something about the CSV data:")

            if user_input:
                with st.spinner("Thinking..."):
                    try:
                        # Append the user message to the history first
                        st.session_state.chat_history.append(("You", user_input))
                        
                        # Run the agent without the unsupported 'memory' keyword argument
                        response = st.session_state.agent.run(user_input)
                        
                        st.session_state.chat_history.append(("Assistant", response))
                        
                        # Rerun the app to show the new messages
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("Chat functionality is disabled because the CSV file was not found.")

if __name__ == "__main__":
    main()


