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
    st.error(f"Missing secret(s): {', '.join(missing_secrets)}. Please set them in your Streamlit secrets.")
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
    json_str = json_str.replace("\\n", "\n")
    json_str = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', json_str)
    return json_str

# --- Setup Google Cloud credentials ---
credentials = None
if "google_credentials" in st.secrets:
    try:
        creds_raw = st.secrets["google_credentials"]
        print(f"Credential string starts with: {creds_raw[:20]}..." if isinstance(creds_raw, str) else "Credentials are not a string")
        if isinstance(creds_raw, str):
            creds_raw = clean_json_string(creds_raw)
            try:
                creds_dict = json.loads(creds_raw)
                required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email"]
                missing_fields = [field for field in required_fields if field not in creds_dict]
                if missing_fields:
                    st.error(f"Google credentials missing required fields: {missing_fields}")
                    print(f"Missing credential fields: {missing_fields}")
                else:
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    st.success("Google Cloud credentials loaded successfully. Voice features are available.")
            except json.JSONDecodeError as json_err:
                st.error(f"Invalid JSON format in google_credentials: {json_err}")
                print(f"JSON parse error: {json_err}")
                print(f"First 100 chars of cleaned JSON: {creds_raw[:100]}...")
        else:
            credentials = service_account.Credentials.from_service_account_info(creds_raw)
            st.success("Google Cloud credentials loaded successfully. Voice features are available.")
    except Exception as e:
        st.warning(f"Error loading Google Cloud credentials: {type(e).__name__}: {str(e)}")
        print(f"Credential error details: {str(e)}")

def get_autoplay_audio_html(audio_bytes, mime_type):
    if audio_bytes is None:
        return ""
    b64 = base64.b64encode(audio_bytes).decode()
    # Use a unique id to force re-render and auto-play
    element_id = f"audio_{int(time.time()*1000)}"
    html_str = f'''
    <audio id="{element_id}" controls autoplay>
        <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
        Your browser does not support the audio element.
    </audio>
    <script>
        var audioElem = document.getElementById("{element_id}");
        if(audioElem) {{
            audioElem.play();
        }}
    </script>
    '''
    return html_str

def send_email(transcript_md):
    subject = "Interview Transcript"
    body = "Please find attached the interview transcript."
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    attachment_part = MIMEText(transcript_md, "plain")
    attachment_part.add_header("Content-Disposition", "attachment", filename="interview_transcript.md")
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
    """Transcribe audio using Google Cloud Speech-to-Text with mono conversion at 16,000 Hz."""
    if credentials is None:
        st.warning("Speech-to-text unavailable. Please type your response instead.")
        return "Voice transcription unavailable. Please type your response."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name
    try:
        print(f"Audio file size: {len(audio_bytes)} bytes")
        # Load audio and force conversion to mono and 16,000 Hz sample rate
        sound = AudioSegment.from_file(tmp_file_path, format="wav")
        mono_audio = sound.set_channels(1).set_frame_rate(16000)
        buffer = io.BytesIO()
        mono_audio.export(buffer, format="wav")
        buffer.seek(0)
        mono_content = buffer.read()
        client = speech.SpeechClient(credentials=credentials)
        audio = speech.RecognitionAudio(content=mono_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-GB",
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True
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

def debug_print_tts(audio_content, label="TTS debug"):
    if audio_content is None:
        print(f"{label}: No audio returned (None).")
        return "0 bytes (No audio data)"
    length = len(audio_content)
    print(f"{label}: returned {length} bytes of audio.")
    return f"{length} bytes of audio."

def text_to_speech(text):
    if credentials is None:
        return None, None
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    formats_to_try = [
        (texttospeech.AudioEncoding.LINEAR16, "audio/wav"),
        (texttospeech.AudioEncoding.MP3, "audio/mp3"),
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
                print(f"TTS: Generated {len(response.audio_content)} bytes of {mime_type} audio.")
                return response.audio_content, mime_type
        except Exception as e:
            print(f"TTS error with {mime_type}: {str(e)}")
    return None, None

