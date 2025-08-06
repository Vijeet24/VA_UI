from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI as LangchainOpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import io
from gtts import gTTS
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder

def load_csv_data(path):
    """Loads data from a local CSV file."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def initialize_agent_and_history(csv_path):
    """Initializes the LangChain agent and chat history in Streamlit's session state."""
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

def main():
    """The main function for the Streamlit application."""
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
            # Display chat history
            for sender, message in st.session_state.chat_history:
                with st.chat_message(name=sender, avatar="🧑" if sender == "You" else "🤖"):
                    st.markdown(message)

            # Use audio_recorder with automatic stop parameters
            st.markdown("---")
            st.write("Click to start recording your question. Recording will stop automatically after a pause.")
            
            # Key change: store audio_bytes in a dedicated session state variable
            audio_bytes = audio_recorder(
                text="Record Audio",
                energy_threshold=(-1.0, 1.0),
                pause_threshold=2.0,
                key="audio_recorder_widget"
            )

            # --- New Logic Flow ---
            
            # Step 1: Transcribe the audio if a new recording is present
            if audio_bytes:
                with st.spinner("Transcribing your audio..."):
                    try:
                        client = st.session_state.get('openai_client')
                        if not client:
                            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            st.session_state.openai_client = client

                        audio_file = io.BytesIO(audio_bytes)
                        audio_file.name = "user_audio.wav"
                        
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=audio_file
                        )
                        user_input = transcript.text
                        st.session_state.chat_history.append(("You", user_input))
                        
                        # --- Run agent and get response ---
                        with st.spinner("Thinking..."):
                            response = st.session_state.agent.invoke({"input": user_input})
                            response_text = response['output']
                            
                            st.session_state.chat_history.append(("Assistant", response_text))
                            
                            # Generate and store audio bytes for the new response
                            tts = gTTS(response_text, lang='en')
                            audio_bytes_io = io.BytesIO()
                            tts.write_to_fp(audio_bytes_io)
                            audio_bytes_io.seek(0)
                            st.session_state.last_audio_bytes = audio_bytes_io.getvalue()
                            
                        # --- End of agent response ---
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Step 2: Play the audio on the final rerun after all messages are in state
            if 'last_audio_bytes' in st.session_state and st.session_state.last_audio_bytes:
                # Use a dummy chat message to hold the audio player visually
                with st.chat_message("Assistant"):
                    st.audio(st.session_state.last_audio_bytes, format='audio/mp3', autoplay=True, loop=False)
                st.session_state.last_audio_bytes = None  # Clear the audio after playing to prevent repetition

        else:
            st.warning("Chat functionality is disabled because the CSV file was not found.")

if __name__ == "__main__":
    main()







