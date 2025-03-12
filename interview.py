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
        # Print first few characters to debug (don't print the whole thing for security)
        print(f"Credential string starts with: {creds_raw[:20]}..." if isinstance(creds_raw, str) else "Credentials are not a string")
        
        if isinstance(creds_raw, str):
            creds_raw = clean_json_string(creds_raw)
            try:
                creds_dict = json.loads(creds_raw)
                # Check for required fields in service account credentials
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

def verify_google_apis():
    """Verify that the required Google APIs are enabled and accessible."""
    if credentials is None:
        return "No credentials available"
    
    results = []
    # Test Text-to-Speech
    try:
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        voices = tts_client.list_voices()
        results.append("✅ Text-to-Speech API is accessible")
    except Exception as e:
        results.append(f"❌ Text-to-Speech API error: {str(e)}")
    
    # Test Speech-to-Text
    try:
        speech_client = speech.SpeechClient(credentials=credentials)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            sample_rate_hertz=16000,
            language_code="en-GB",
        )
        results.append("✅ Speech-to-Text API is accessible")
    except Exception as e:
        results.append(f"❌ Speech-to-Text API error: {str(e)}")
    
    return "\n".join(results)

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
    if credentials is None:
        st.warning("Speech-to-text unavailable. Please type your response instead.")
        return "Voice transcription unavailable. Please type your response."
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        # Convert stereo to mono
        try:
            sound = AudioSegment.from_file(tmp_file_path, format="wav")
            sound_mono = sound.set_channels(1)
            sound_mono.export(tmp_file_path, format="wav")
        except Exception as conv_err:
            raise Exception(f"Error converting audio to mono: {conv_err}")
        
        client = speech.SpeechClient(credentials=credentials)
        
        with open(tmp_file_path, "rb") as audio_file:
            content = audio_file.read()
        
        # Attempt 1: LINEAR16
        try:
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
        except Exception as first_attempt_error:
            print(f"First transcription attempt failed: {first_attempt_error}")
        
        # Attempt 2: let the API detect encoding
        try:
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
                sample_rate_hertz=48000,
                language_code="en-GB",
            )
            response = client.recognize(config=config, audio=audio)
            
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript
            return transcript.strip()
        except Exception as second_attempt_error:
            raise Exception(f"Multiple transcription attempts failed. Last error: {second_attempt_error}")
            
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
    """Try Google TTS with a British English voice. If the first voice fails, fallback."""
    if credentials is None:
        return None
    
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    
    # Simplify the voice selection - use standard voice without specifying a name
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    
    # Use MP3 format for better compatibility
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,  # Normal speaking rate
        pitch=0.0,  # Default pitch
        volume_gain_db=0.0  # Default volume
    )
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        if response.audio_content:
            print(f"TTS: Generated {len(response.audio_content)} bytes of audio.")
            return response.audio_content
        else:
            print("TTS: Received empty audio content.")
            return None
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return None

def test_tts_detailed():
    """Run a detailed test of TTS functionality with debugging information."""
    if credentials is None:
        return "No credentials available", None
    
    try:
        # 1. Create client
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        print("TTS client created successfully")
        
        # 2. List available voices to verify API connection
        try:
            voices = tts_client.list_voices(language_code="en-GB")
            voice_names = [voice.name for voice in voices.voices]
            print(f"Available voices: {voice_names[:3]} (showing first 3 of {len(voice_names)})")
        except Exception as e:
            print(f"Error listing voices: {str(e)}")
        
        # 3. Test with minimal voice settings
        test_input = texttospeech.SynthesisInput(text="This is a test of text to speech.")
        voice = texttospeech.VoiceSelectionParams(language_code="en-GB")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        response = tts_client.synthesize_speech(
            input=test_input, 
            voice=voice,
            audio_config=audio_config
        )
        
        print(f"TTS response received with {len(response.audio_content)} bytes")
        return "TTS test completed successfully", response.audio_content
    except Exception as e:
        error_message = f"TTS test failed: {type(e).__name__}: {str(e)}"
        print(error_message)
        return error_message, None

def try_all_tts_voices():
    """Try all available TTS voices to find one that works."""
    if credentials is None:
        return "No credentials available", None
    
    results = []
    try:
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        voices = tts_client.list_voices(language_code="en-GB")
        
        test_text = "This is a test of the Google Text-to-Speech API."
        test_input = texttospeech.SynthesisInput(text=test_text)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        # Try up to 5 voices
        for i, voice_info in enumerate(voices.voices[:5]):
            try:
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-GB",
                    name=voice_info.name
                )
                
                print(f"Trying voice: {voice_info.name}")
                response = tts_client.synthesize_speech(
                    input=test_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                if response and response.audio_content:
                    results.append(f"✅ Voice {voice_info.name}: Generated {len(response.audio_content)} bytes")
                    # Return the first successful voice
                    return f"Found working voice: {voice_info.name}", response.audio_content
                else:
                    results.append(f"❌ Voice {voice_info.name}: No audio content returned")
            except Exception as e:
                results.append(f"❌ Voice {voice_info.name}: Error: {str(e)}")
        
        # If we get here, none of the voices worked
        return "No working voices found. Details:\n" + "\n".join(results), None
    except Exception as e:
        error_message = f"Error testing voices: {type(e).__name__}: {str(e)}"
        print(error_message)
        return error_message, None

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

    # --- Verify Google APIs button ---
    st.write("---")
    st.write("### Verify Google Cloud APIs")
    if st.button("Check API Access"):
        with st.spinner("Verifying API access..."):
            results = verify_google_apis()
            st.code(results, language="text")

    # --- Minimal TTS Test button ---
    st.write("---")
    st.write("### Test Basic TTS")
    st.write("Use this button to confirm if TTS is working at all.")
    if st.button("Test Basic TTS"):
        if credentials:
            st.write("Attempting a minimal TTS call...")
            tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            test_input = texttospeech.SynthesisInput(text="Hello from the TTS test!")
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-GB"  # no named voice
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

            try:
                response = tts_client.synthesize_speech(
                    input=test_input,
                    voice=voice,
                    audio_config=audio_config
                )
                audio_bytes = response.audio_content
                bytes_len = len(audio_bytes) if audio_bytes else 0
                st.write(f"Basic TTS returned {bytes_len} bytes.")
                print(f"[Minimal TTS] returned {bytes_len} bytes.")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.error(f"TTS minimal test failed: {e}")
                print(f"TTS minimal test error: {e}")
        else:
            st.error("No credentials available for TTS test.")

    # --- Detailed TTS Test button ---
    st.write("### Detailed TTS Test")
    st.write("This test will provide more information to diagnose TTS issues.")
    if st.button("Run Detailed TTS Test"):
        if credentials:
            with st.spinner("Testing TTS with detailed diagnostics..."):
                message, audio_bytes = test_tts_detailed()
                st.write(message)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success(f"Successfully generated {len(audio_bytes)} bytes of audio.")
                else:
                    st.error("No audio was generated.")
        else:
            st.error("No credentials available for TTS test.")

    # --- Try All Voices button ---
    st.write("### Try All Available Voices")
    st.write("This will test multiple voices to find one that works.")
    if st.button("Try All Voices"):
        if credentials:
            with st.spinner("Testing multiple TTS voices..."):
                message, audio_bytes = try_all_tts_voices()
                st.write(message)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success(f"Successfully generated {len(audio_bytes)} bytes of audio.")
                else:
                    st.error("No audio was generated from any voice.")
        else:
            st.error("No credentials available for voice testing.")
    
    st.write("---")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = (
            "Thank you for agreeing to speak with us about your recent rugby taster session. "
            "To begin, can you tell me a bit about yourself and any previous experience with rugby or other sports?"
        )
        # Generate TTS for the initial question if credentials are available
        if credentials is not None:
            st.write("Attempting TTS for initial question...")
            try:
                audio_bytes = text_to_speech(st.session_state.current_question)
                info_str = debug_print_tts(audio_bytes, label="Initial Q TTS")
                if audio_bytes and len(audio_bytes) > 0:
                    st.session_state.current_audio = audio_bytes
                    st.success("Initial question audio generated successfully.")
                else:
                    st.warning(f"No audio returned for the initial question. Debug: {info_str}")
                    st.session_state.current_audio = None
            except Exception as e:
                st.warning(f"Unable to generate speech for initial question: {e}")
                st.session_state.current_audio = None

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
        st.markdown(f"**AI Question:** {st.session_state.current_question}")
        
        # If we have TTS audio, show it
        if "current_audio" in st.session_state and st.session_state.current_audio:
            st.caption(f"(Audio length = {len(st.session_state.current_audio)} bytes)")
            st.audio(st.session_state.current_audio, format="audio/mp3")
            if st.button("Regenerate Question Audio"):
                try:
                    audio_bytes = text_to_speech(st.session_state.current_question)
                    dbg = debug_print_tts(audio_bytes, label="Regenerated Q TTS")
                    if audio_bytes and len(audio_bytes) > 0:
                        st.session_state.current_audio = audio_bytes
                        st.success(f"Audio regenerated successfully. {dbg}")
                    else:
                        st.warning(f"No audio returned. {dbg}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error regenerating audio: {e}")
        else:
            st.warning("Audio for this question is not available (possibly TTS returned 0 bytes).")
            if st.button("Generate Audio for Question"):
                try:
                    audio_bytes = text_to_speech(st.session_state.current_question)
                    dbg = debug_print_tts(audio_bytes, label="On-demand Q TTS")
                    if audio_bytes and len(audio_bytes) > 0:
                        st.session_state.current_audio = audio_bytes
                        st.success(f"Audio generated successfully. {dbg}")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to generate audio (0 bytes). {dbg}")
                except Exception as e:
                    st.error(f"Error generating audio: {e}")
        
        if credentials is not None:
            st.write("**Speak your answer:**")
            audio_bytes = audio_recorder()
            st.warning("Note: Voice recognition may not be perfect. If your response is off, please type below.")
            if audio_bytes:
                st.success("Audio recorded! Transcribing...")
                try:
                    transcript = transcribe_audio(audio_bytes)
                    st.session_state.current_transcript = transcript
                    st.write(f"**Transcribed:** {transcript}")
                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")
                    st.session_state.current_transcript = ""
        else:
            st.info("Voice recording not available (no credentials). Please type your response below.")
            st.session_state.current_transcript = ""
        
        user_answer = st.text_area(
            "Your response (edit transcription or type):", 
            value=st.session_state.get("current_transcript", ""),
            key=f"user_input_{len(st.session_state.conversation)}"
        )

        user_rating = st.radio(
            "On a scale of 1–10, how would you rate your experience relating to the current question/topic?",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=4
        )

        # Progress
        completed_questions = len([entry for entry in st.session_state.conversation if entry['role'] == "user"])
        progress_percentage = min(completed_questions / total_questions, 1.0)
        st.write(f"**Interview Progress: {completed_questions} out of {total_questions} questions answered**")
        st.progress(progress_percentage)

        if st.button("Submit Answer"):
            if user_answer.strip():
                combined_user_content = f"Answer: {user_answer}\nRating: {user_rating}"
                st.session_state.conversation.append({"role": "user", "content": combined_user_content})
                
                ai_prompt = (
                    f"User's answer: {user_answer}\n"
                    f"User's rating: {user_rating}\n"
                    f"Provide feedback and ask a follow-up question."
                )
                ai_response = generate_response(ai_prompt, st.session_state.conversation)
                
                st.session_state.conversation.append({"role": "assistant", "content": ai_response})
                st.session_state.current_question = ai_response
                
                # Attempt TTS for the new question
                if credentials is not None:
                    try:
                        next_audio = text_to_speech(ai_response)
                        dbg = debug_print_tts(next_audio, label="Follow-up TTS")
                        if next_audio and len(next_audio) > 0:
                            st.session_state.current_audio = next_audio
                        else:
                            st.warning(f"Failed to generate audio for follow-up. {dbg}")
                            st.session_state.current_audio = None
                    except Exception as e:
                        st.warning(f"Unable to generate speech: {e}")
                        st.session_state.current_audio = None
                
                st.session_state.current_transcript = ""
                st.rerun()
            else:
                st.warning("Please provide an answer before submitting.")

        if st.button("End Interview"):
            st.success("Interview completed! Thank you for sharing your rugby taster session experience.")
            st.session_state.current_question = "Interview ended"
            
            transcript_md = convert_to_markdown(st.session_state.conversation)
            if send_email(transcript_md):
                st.info("Your transcript has been emailed to the researcher.")
            
            st.markdown(get_transcript_download_link(st.session_state.conversation), unsafe_allow_html=True)

        if st.checkbox("Show Interview Transcript"):
            st.write("**Interview Transcript:**")
            for entry in st.session_state.conversation:
                st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")
                st.write("---")

        if st.button("Restart Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
