import streamlit as st
from openai import OpenAI
import pandas as pd
import base64
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tempfile
import os
import json
import re
import io
import numpy as np
import time

# For speech recognition and TTS
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
from google.oauth2 import service_account

# For audio recording
from audio_recorder_streamlit import audio_recorder

# pydub for audio conversion
import warnings
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*", category=SyntaxWarning)
from pydub import AudioSegment

# --- Retrieve secrets (with fallback to None) ---
PASSWORD = st.secrets.get("password", None)
OPENAI_API_KEY = st.secrets.get("openai_api_key", None)
SENDER_EMAIL = st.secrets.get("sender_email", None)
EMAIL_PASSWORD = st.secrets.get("email_password", None)
RECEIVER_EMAIL = "tony.myers@staff.newman.ac.uk"

# If any secrets are missing, stop the app and show an error
missing_secrets = []
if not PASSWORD:
    missing_secrets.append("password")
if not OPENAI_API_KEY:
    missing_secrets.append("openai_api_key")
if not SENDER_EMAIL:
    missing_secrets.append("sender_email")
if not EMAIL_PASSWORD:
    missing_secrets.append("email_password")

if missing_secrets:
    st.error(
        f"Missing secret(s): {', '.join(missing_secrets)}. "
        "Please set them in your Streamlit secrets."
    )
    st.stop()

# --- Interview topics ---
interview_topics = [
    "Introduction and background in relation to rugby",
    "Motivation for attending the taster session",
    "Overall experience and atmosphere of the session",
    "Most enjoyable parts of the session",
    "Challenges or difficulties faced",
    "Rating aspect: enjoyment level (1–10)",
    "Perceived impact on willingness to continue with rugby"
]
total_questions = len(interview_topics)

def clean_json_string(json_str):
    """Clean a JSON string that may contain invalid control characters."""
    # Replace literal \n with actual newlines (helps with private key formatting)
    json_str = json_str.replace("\\n", "\n")
    
    # Remove any non-printable characters except valid whitespace
    json_str = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', json_str)
    
    return json_str

# --- Setup Google Cloud credentials ---
credentials = None
if "google_credentials" in st.secrets:
    try:
        creds_raw = st.secrets["google_credentials"]
        
        if isinstance(creds_raw, str):
            creds_raw = clean_json_string(creds_raw)
            try:
                creds_dict = json.loads(creds_raw)
                # Check for required fields in service account credentials
                required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email"]
                missing_fields = [field for field in required_fields if field not in creds_dict]
                if missing_fields:
                    st.error(f"Google credentials missing required fields: {missing_fields}")
                else:
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
            except json.JSONDecodeError as json_err:
                st.error(f"Invalid JSON format in google_credentials: {json_err}")
        else:
            credentials = service_account.Credentials.from_service_account_info(creds_raw)
    except Exception as e:
        st.warning(f"Error loading Google Cloud credentials: {type(e).__name__}: {str(e)}")

def generate_response(prompt, conversation_history=None):
    try:
        if conversation_history is None:
            conversation_history = []

        system_content = """You are an experienced and considerate interviewer focusing on young people's experiences with rugby taster sessions aimed at diversifying the participation base. Use British English in your responses (e.g., 'democratised'). 
Ensure your responses are complete and not truncated. After each user response, provide brief feedback and ask a relevant follow-up question based on their answer. Tailor your questions to the user's previous responses, avoiding repetition and exploring areas they haven't covered. Be adaptive and create a natural flow of conversation."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "system", "content": f"Interview topics: {interview_topics}"},
            *conversation_history[-6:],  # last 6 exchanges for context
            {"role": "user", "content": prompt}
        ]

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=110,
            n=1,
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred in generate_response: {str(e)}"

def convert_to_markdown(conversation):
    md_text = "# Rugby Taster Session Interview Transcript\n\n"
    for entry in conversation:
        role = entry["role"].capitalize()
        content = entry["content"]
        md_text += f"**{role}**: {content}\n\n---\n\n"
    return md_text

def get_transcript_download_link(conversation):
    md_text = convert_to_markdown(conversation)
    b64 = base64.b64encode(md_text.encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="interview_transcript.md">Download Transcript</a>'
    return href

def get_audio_download_link(audio_bytes, mime_type, filename="audio.mp3"):
    """Generate a download link for the audio file."""
    if audio_bytes is None:
        return ""
    
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download Audio</a>'
    return href

def send_email(transcript_md):
    subject = "Interview Transcript"
    body = "Please find attached the interview transcript."
    
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    attachment_part = MIMEText(transcript_md, "plain")
    attachment_part.add_header(
        "Content-Disposition",
        "attachment",
        filename="interview_transcript.md"
    )
    message.attach(attachment_part)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def transcribe_audio(audio_bytes):
    """Basic voice transcription function that returns input directly to Google Speech API."""
    if credentials is None:
        st.warning("Speech-to-text unavailable. Please type your response instead.")
        return "Voice transcription unavailable. Please type your response."
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        client = speech.SpeechClient(credentials=credentials)
        
        with open(tmp_file_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code="en-GB",
        )
        
        response = client.recognize(config=config, audio=audio)
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        if transcript.strip():
            return transcript.strip()
        else:
            return "Could not transcribe audio. Please type your response."
            
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return f"Error: {str(e)}"
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def text_to_speech(text):
    """Generate TTS audio with fallback formats."""
    if credentials is None:
        return None, None
    
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    
    # Simple voice selection
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    
    # Try different audio formats in order of preference
    formats_to_try = [
        (texttospeech.AudioEncoding.MP3, "audio/mp3"),
        (texttospeech.AudioEncoding.LINEAR16, "audio/wav"),
        (texttospeech.AudioEncoding.OGG_OPUS, "audio/ogg")
    ]
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    for encoding, mime_type in formats_to_try:
        try:
            audio_config = texttospeech.AudioConfig(
                audio_encoding=encoding,
                speaking_rate=1.0,
                pitch=0.0,
                volume_gain_db=0.0
            )
            
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            if response.audio_content and len(response.audio_content) > 0:
                return response.audio_content, mime_type
        except Exception as e:
            print(f"TTS error with {mime_type}: {str(e)}")
    
    return None, None

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter password to access the interview app:", type="password")
        if st.button("Submit"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.success("Access granted.")
            else:
                st.error("Incorrect password.")
        return  # Stop if not authenticated

    # Main content
    st.title("Rugby Taster Session Voice Interview Bot")

    if credentials is None:
        st.warning("Google Cloud Speech services are not configured. Voice features will not be available.")

    # Initialize conversation and question state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = (
            "Thank you for agreeing to speak with us about your recent rugby taster session. "
            "To begin, can you tell me a bit about yourself and any previous experience with rugby or other sports?"
        )
        # Generate TTS for the initial question if credentials are available
        if credentials is not None:
            try:
                audio_bytes, mime_type = text_to_speech(st.session_state.current_question)
                if audio_bytes and len(audio_bytes) > 0:
                    st.session_state.current_audio = audio_bytes
                    st.session_state.current_audio_mime = mime_type
                else:
                    st.session_state.current_audio = None
                    st.session_state.current_audio_mime = None
            except Exception as e:
                print(f"Initial TTS error: {e}")
                st.session_state.current_audio = None
                st.session_state.current_audio_mime = None

    # Initialize transcript state
    if "current_transcript" not in st.session_state:
        st.session_state.current_transcript = ""
    
    st.write("""
    **Information Sheet and Consent**  
    By ticking yes below, you consent to participate in this interview about your experience in a rugby taster session. 
    Your responses may be anonymously quoted in publications. You may end the interview at any time and request 
    your data be removed by emailing tony.myers@staff.newman.ac.uk. 
    An AI assistant will ask main questions and follow-up probing questions.
    """)

    consent = st.checkbox("I have read the information sheet and give my consent to participate in this interview.")

    if consent:
        # Display current question
        st.subheader("AI Question:")
        st.write(st.session_state.current_question)
        
        # Display audio if available - using standard audio player
        if "current_audio" in st.session_state and st.session_state.current_audio:
            mime_type = st.session_state.get("current_audio_mime", "audio/mp3")
            st.audio(st.session_state.current_audio, format=mime_type)
        
        st.write("---")
        
        # Voice recording section - using original code that was working
        if credentials is not None:
            st.subheader("Your Response:")
            
            # Larger, more visible recording button
            st.write("**Record your answer:**")
            
            # Using the audio recorder with original settings
            audio_bytes = audio_recorder()
            
            # When audio is recorded
            if audio_bytes:
                with st.spinner("Transcribing your response..."):
                    # Use original transcription function that was working
                    transcript = transcribe_audio(audio_bytes)
                    st.session_state.current_transcript = transcript
                
                st.write("**Transcribed text:**")
                st.markdown(f"_{transcript}_")
        else:
            st.info("Voice recording not available. Please type your response below.")
        
        # Text area for editing or typing response
        user_answer = st.text_area(
            "Edit transcription or type your response:", 
            value=st.session_state.current_transcript,
            height=150,
            key=f"user_input_{len(st.session_state.conversation)}"
        )

        # User rating
        user_rating = st.radio(
            "On a scale of 1–10, how would you rate your experience relating to the current question/topic?",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=4,
            horizontal=True
        )

        # Progress indicator
        completed_questions = len([entry for entry in st.session_state.conversation if entry['role'] == "user"])
        progress_percentage = min(completed_questions / total_questions, 1.0)
        st.write(f"**Interview Progress: {completed_questions} out of {total_questions} questions answered**")
        st.progress(progress_percentage)

        # Submit button
        if st.button("Submit Answer", type="primary"):
            if user_answer.strip():
                # Add user response to conversation
                combined_user_content = f"Answer: {user_answer}\nRating: {user_rating}"
                st.session_state.conversation.append({"role": "user", "content": combined_user_content})
                
                # Generate AI response
                with st.spinner("Generating follow-up question..."):
                    ai_prompt = (
                        f"User's answer: {user_answer}\n"
                        f"User's rating: {user_rating}\n"
                        f"Provide feedback and ask a follow-up question."
                    )
                    ai_response = generate_response(ai_prompt, st.session_state.conversation)
                    
                    # Add AI response to conversation
                    st.session_state.conversation.append({"role": "assistant", "content": ai_response})
                    st.session_state.current_question = ai_response
                
                # Generate audio for the new question
                if credentials is not None:
                    try:
                        next_audio, mime_type = text_to_speech(ai_response)
                        if next_audio and len(next_audio) > 0:
                            st.session_state.current_audio = next_audio
                            st.session_state.current_audio_mime = mime_type
                        else:
                            st.session_state.current_audio = None
                            st.session_state.current_audio_mime = None
                    except Exception as e:
                        print(f"TTS error for follow-up: {e}")
                        st.session_state.current_audio = None
                        st.session_state.current_audio_mime = None
                
                # Reset transcript
                st.session_state.current_transcript = ""
                
                # Rerun to show next question
                st.rerun()
            else:
                st.warning("Please provide an answer before submitting.")

        # End interview button
        if st.button("End Interview"):
            st.success("Interview completed! Thank you for sharing your rugby taster session experience.")
            st.session_state.current_question = "Interview ended"
            
            # Email and download transcript
            transcript_md = convert_to_markdown(st.session_state.conversation)
            if send_email(transcript_md):
                st.info("Your transcript has been emailed to the researcher.")
            
            st.markdown(get_transcript_download_link(st.session_state.conversation), unsafe_allow_html=True)

        # View transcript
        with st.expander("Show Interview Transcript", expanded=False):
            st.write("**Interview Transcript:**")
            for entry in st.session_state.conversation:
                st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")
                st.write("---")

        # Restart button
        if st.button("Restart Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
