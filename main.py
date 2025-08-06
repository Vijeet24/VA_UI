from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI as LangchainOpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
from gtts import gTTS
import base64
from streamlit_mic_recorder import mic_recorder

def load_csv_data(path):
    """
    Loads data from a local CSV file.
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

def text_to_audio(text):
    """
    Converts text to speech and plays it.
    """
    try:
        tts = gTTS(text, lang='en')
        tts.save("response.mp3")
        with open("response.mp3", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format='audio/mp3', autoplay=True, loop=False)
    except Exception as e:
        st.warning(f"Text-to-speech failed: {e}")

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
        
        df = load_csv_data(local_csv_path)
        if df is not None:
            st.dataframe(df)

        if st.button("Refresh Data"):
            st.rerun()

    # Right: Continuous Chat
    with col2:
        st.subheader("💬 Continuous Chat with CSV")

        if st.session_state.agent is not None:
            # Create a placeholder for the chat messages
            chat_container = st.container()

            with chat_container:
                for sender, message in st.session_state.chat_history:
                    with st.chat_message(name=sender, avatar="🧑" if sender == "You" else "🤖"):
                        st.markdown(message)
                
            # Audio input and chat logic
            st.markdown("---")
            st.write("Click to record your question:")
            
            audio_input = mic_recorder(start_prompt="⏺️ Start recording", stop_prompt="⏹️ Stop recording", key="recorder")

            if audio_input:
                try:
                    # Speech-to-Text with OpenAI's Whisper API
                    # Note: You need the 'openai' package for this.
                    client = st.session_state.get('openai_client')
                    if not client:
                        from openai import OpenAI
                        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        st.session_state.openai_client = client

                    with st.spinner("Transcribing your audio..."):
                        # Save the recorded audio to a file
                        with open("user_audio.wav", "wb") as f:
                            f.write(audio_input['bytes'])
                        
                        audio_file = open("user_audio.wav", "rb")
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=audio_file
                        )
                        user_input = transcript.text
                        st.session_state.chat_history.append(("You", user_input))
                        st.rerun()

                except Exception as e:
                    st.error(f"Error transcribing audio: {e}")

            # This part handles the chatbot response
            if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "You":
                user_input = st.session_state.chat_history[-1][1]
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.agent.run(user_input)
                        st.session_state.chat_history.append(("Assistant", response))
                        
                        # Text-to-Speech for the response
                        text_to_audio(response)
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("Chat functionality is disabled because the CSV file was not found.")

if __name__ == "__main__":
    main()



