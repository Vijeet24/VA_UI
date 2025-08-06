from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI as LangchainOpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import io
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

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
    Converts text to speech and plays it using an in-memory buffer.
    This fixes the MediaFileStorageError by not relying on a temporary file.
    """
    try:
        tts = gTTS(text, lang='en')
        # Use a BytesIO buffer to store the audio in memory
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        st.audio(audio_bytes_io, format='audio/mp3', autoplay=True, loop=False)
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

    if 'agent' not in st.session_state:
        initialize_agent_and_history(local_csv_path)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Live CSV Data")
        
        df = load_csv_data(local_csv_path)
        if df is not None:
            st.dataframe(df)

        if st.button("Refresh Data"):
            st.rerun()

    with col2:
        st.subheader("💬 Continuous Chat with CSV")

        if st.session_state.agent is not None:
            chat_container = st.container()
            with chat_container:
                for sender, message in st.session_state.chat_history:
                    with st.chat_message(name=sender, avatar="🧑" if sender == "You" else "🤖"):
                        st.markdown(message)
            
            st.markdown("---")
            st.write("Click to record your question:")
            
            audio_input = mic_recorder(start_prompt="⏺️ Start recording", stop_prompt="⏹️ Stop recording", just_once=True, key="recorder")

            if 'new_audio_transcription' not in st.session_state:
                st.session_state.new_audio_transcription = None

            # Step 1: Transcribe the audio and store the result
            if audio_input and st.session_state.new_audio_transcription is None:
                try:
                    with st.spinner("Transcribing your audio..."):
                        client = st.session_state.get('openai_client')
                        if not client:
                            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            st.session_state.openai_client = client

                        audio_file = io.BytesIO(audio_input['bytes'])
                        audio_file.name = "user_audio.wav"
                        
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=audio_file
                        )
                        st.session_state.new_audio_transcription = transcript.text
                        st.rerun()
                except Exception as e:
                    st.error(f"Error transcribing audio: {e}")

            # Step 2: Run the agent with the transcribed text and get a response
            if st.session_state.new_audio_transcription:
                user_input = st.session_state.new_audio_transcription
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.new_audio_transcription = None

                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.agent.invoke({"input": user_input})
                        
                        st.session_state.chat_history.append(("Assistant", response['output']))
                        
                        # --- MODIFIED AUDIO LOGIC ---
                        # Use a dedicated session state variable to hold the audio bytes
                        tts = gTTS(response['output'], lang='en')
                        audio_bytes_io = io.BytesIO()
                        tts.write_to_fp(audio_bytes_io)
                        audio_bytes_io.seek(0)
                        st.session_state.audio_bytes = audio_bytes_io.getvalue()
                        # --- END MODIFIED AUDIO LOGIC ---

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            # --- NEW: Play the audio on the final rerun ---
            if 'audio_bytes' in st.session_state and st.session_state.audio_bytes:
                st.audio(st.session_state.audio_bytes, format='audio/mp3', autoplay=True, loop=False)
                st.session_state.audio_bytes = None  # Clear the audio after playing
            # --- END NEW LOGIC ---

        else:
            st.warning("Chat functionality is disabled because the CSV file was not found.")

if __name__ == "__main__":
    main()








