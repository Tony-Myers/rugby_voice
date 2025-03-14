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

# Check for missing secrets
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
    element_id = f"audio_{int(time.time()*1000)}"
    html_str = f'''
    <audio id="{element_id}" controls autoplay>
        <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
        Your browser does not support the audio element.
    </audio>
    <script>
        setTimeout(function() {{
            var audioElem = document.getElementById("{element_id}");
            if(audioElem) {{
                audioElem.play().catch(function(e) {{
                    console.log("Autoplay failed:", e);
                }});
            }}
        }}, 500);
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
        # Convert to mono and enforce 16,000 Hz sample rate
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

def generate_response(prompt, conversation_history=None):
    try:
        if conversation_history is None:
            conversation_history = []
        system_content = (
            "You are an experienced and considerate interviewer focusing on young people's experiences with rugby taster sessions aimed at diversifying the participation base. "
            "Use British English in your responses (e.g., 'democratised'). "
            "Ensure your responses are complete and not truncated. After each user response, provide brief feedback and ask a relevant follow-up question based on their answer. "
            "Tailor your questions to the user's previous responses, avoiding repetition and exploring areas they have not covered. Be adaptive and create a natural flow of conversation."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "system", "content": f"Interview topics: {interview_topics}"},
            *conversation_history[-6:],
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

def main():
    # Authentication block
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        password = st.text_input("Enter password to access the interview app:", type="password")
        if st.button("Submit"):
            if password == PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Access granted.")
            else:
                st.error("Incorrect password.")
        st.stop()

    # Consent page
    if "consent_obtained" not in st.session_state:
        st.session_state["consent_obtained"] = False

    if not st.session_state["consent_obtained"]:
        st.title("Information Sheet and Consent")
        st.write("""
        **Information Sheet and Consent** 

        By ticking "Yes" below, you consent to participate in this interview about your experience in a rugby taster session. 
        Your responses may be anonymously quoted in publications. You may end the interview at any time and request 
        your data be removed by emailing tony.myers@staff.newman.ac.uk.

        An AI assistant will ask main questions and follow-up probing questions.
        """)
        consent_choice = st.radio("Do you consent to participate?", ("No", "Yes"), index=1)
        if consent_choice == "Yes":
            st.session_state["consent_obtained"] = True
            st.rerun()
        else:
            st.stop()

    st.title("Rugby Taster Session Voice Interview Bot")
    if credentials is None:
        st.warning("Google Cloud Speech services are not configured. Voice features are not available.")

    # We track phases: "ready_for_question", "rating_requested", "interview_over"
    if "phase" not in st.session_state:
        st.session_state["phase"] = "ready_for_question"

    # Holds entire conversation
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    # This will hold the question text and associated audio we want to play:
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = (
            "Thank you for agreeing to speak with us about your recent rugby taster session. "
            "To begin, can you tell me a bit about yourself and any previous experience with rugby or other sports?"
        )
        # Generate TTS for initial question
        if credentials is not None:
            try:
                audio_bytes, mime_type = text_to_speech(st.session_state["current_question"])
                dbg = debug_print_tts(audio_bytes, label="Initial Q TTS")
                if audio_bytes and len(audio_bytes) > 0:
                    st.session_state["current_audio"] = audio_bytes
                    st.session_state["current_audio_mime"] = mime_type
                else:
                    st.warning(f"No audio returned for the initial question. Debug: {dbg}")
                    st.session_state["current_audio"] = None
                    st.session_state["current_audio_mime"] = None
            except Exception as e:
                st.warning(f"Unable to generate speech for initial question: {e}")
                st.session_state["current_audio"] = None
                st.session_state["current_audio_mime"] = None

    # To avoid replaying the same audio on every rerun, store a flag keyed by the question text:
    if "audio_played_for" not in st.session_state:
        st.session_state["audio_played_for"] = ""

    # Display the AI question text
    st.markdown(f"**AI Question:** {st.session_state['current_question']}")

    # Autoplay the question ONLY if we have not played it before
    if st.session_state.get("current_audio") and st.session_state["audio_played_for"] != st.session_state["current_question"]:
        st.components.v1.html(
            get_autoplay_audio_html(st.session_state["current_audio"], st.session_state["current_audio_mime"]),
            height=80
        )
        # Mark that we've played audio for this exact question
        st.session_state["audio_played_for"] = st.session_state["current_question"]

    # Recording section
    if st.session_state["phase"] == "ready_for_question":
        st.write("---")
        st.write("Please allow access to your microphone if prompted.")
        st.write("Press the button below to **start recording** (it turns red). To **stop recording**, press it again.")
        audio_bytes = audio_recorder(
            pause_threshold=9999,
            recording_color="#FF5733",
            neutral_color="#6aa36f",
            text="Press to START or STOP recording",
            energy_threshold=0.01,
        )

        if audio_bytes:
            try:
                recorded_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
                duration_sec = len(recorded_audio) / 1000.0
                if duration_sec < 2.0:
                    st.warning("Recording too short (less than 2 seconds). Please record again.")
                else:
                    st.success("Recording captured! Transcribing now...")
                    with st.spinner("Transcribing your speech..."):
                        transcript = transcribe_audio(audio_bytes)
                        st.write(f"**Transcribed:** {transcript}")

                    # Store user response in conversation
                    st.session_state["conversation"].append({"role": "user", "content": transcript})

                    # Now move to rating phase
                    st.session_state["phase"] = "rating_requested"
                    rating_prompt = "Please provide a rating from 1 to 10 for your experience. Then press the 'Submit Rating' button."
                    audio_rating, rating_mime = text_to_speech(rating_prompt)
                    st.session_state["rating_audio"] = audio_rating
                    st.session_state["rating_mime"] = rating_mime

                    # We will play the rating prompt automatically in that phase, so clear any old question audio
                    st.session_state["current_audio"] = None
                    st.session_state["current_audio_mime"] = None

                    st.rerun()
            except Exception as e:
                st.error(f"Error processing recorded audio: {e}")

    elif st.session_state["phase"] == "rating_requested":
        # If we have rating audio not played yet, auto-play it
        if "rating_audio" in st.session_state and st.session_state["rating_audio"]:
            # Only play once if not already
            if st.session_state["audio_played_for"] != "rating_prompt":
                st.components.v1.html(
                    get_autoplay_audio_html(st.session_state["rating_audio"], st.session_state["rating_mime"]),
                    height=80
                )
                st.session_state["audio_played_for"] = "rating_prompt"

        st.write("Please provide a rating from 1 to 10, then press 'Submit Rating'.")

        if "user_rating" not in st.session_state:
            st.session_state["user_rating"] = 5  # default

        st.session_state["user_rating"] = st.slider("Your rating:", 1, 10, st.session_state["user_rating"])
        if st.button("Submit Rating"):
            # Add rating to conversation
            rating_val = st.session_state["user_rating"]
            st.session_state["conversation"].append(
                {"role": "user", "content": f"Rating provided: {rating_val}"}
            )

            # Generate AI follow-up
            last_answer = st.session_state["conversation"][-2]["content"]  # user's last main response
            ai_prompt = (
                f"User's last answer: {last_answer}\n"
                f"User's rating: {rating_val}\n"
                "Provide follow-up feedback and ask the next question."
            )
            ai_response = generate_response(ai_prompt, st.session_state["conversation"])
            st.session_state["conversation"].append({"role": "assistant", "content": ai_response})

            # Prepare TTS for AI question
            st.session_state["current_question"] = ai_response
            if credentials is not None:
                try:
                    next_audio, mime_type = text_to_speech(ai_response)
                    dbg = debug_print_tts(next_audio, label="Follow-up Q TTS")
                    if next_audio and len(next_audio) > 0:
                        st.session_state["current_audio"] = next_audio
                        st.session_state["current_audio_mime"] = mime_type
                    else:
                        st.warning(f"Failed to generate audio for follow-up question. {dbg}")
                        st.session_state["current_audio"] = None
                        st.session_state["current_audio_mime"] = None
                except Exception as e:
                    st.warning(f"Unable to generate speech for follow-up question: {e}")
                    st.session_state["current_audio"] = None
                    st.session_state["current_audio_mime"] = None

            # Reset so the new question can play once
            st.session_state["audio_played_for"] = ""

            # Go back to "ready_for_question"
            st.session_state["phase"] = "ready_for_question"
            st.rerun()

    # End interview logic
    if st.button("End Interview"):
        st.success("Interview completed! Thank you for sharing your rugby taster session experience.")
        st.session_state["phase"] = "interview_over"
        transcript_md = convert_to_markdown(st.session_state["conversation"])
        if send_email(transcript_md):
            st.info("Your transcript has been emailed to the researcher.")
        st.markdown(get_transcript_download_link(st.session_state["conversation"]), unsafe_allow_html=True)

    if st.checkbox("Show Interview Transcript"):
        st.write("**Interview Transcript:**")
        for entry in st.session_state["conversation"]:
            st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")
            st.write("---")

    if st.session_state["phase"] == "interview_over":
        st.stop()

    if st.button("Restart Interview"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
