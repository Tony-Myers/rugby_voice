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

# For speech recognition and TTS
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
from google.oauth2 import service_account

# For audio recording
from audio_recorder_streamlit import audio_recorder

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

# --- Setup Google Cloud credentials ---
credentials = None
if "google_credentials" in st.secrets:
    try:
        # Get credentials string from secrets
        creds_raw = st.secrets["google_credentials"]
        
        # Clean up the string if needed (remove extra quotes, fix escaping)
        if isinstance(creds_raw, str):
            # Strip any surrounding quotes if present
            creds_raw = creds_raw.strip('"\'')
            
            # Try to parse the JSON
            try:
                creds_dict = json.loads(creds_raw)
                
                # Particularly check for and fix private_key formatting
                if "private_key" in creds_dict and isinstance(creds_dict["private_key"], str):
                    # Ensure private key has proper newlines
                    if "\\n" in creds_dict["private_key"] and "\n" not in creds_dict["private_key"]:
                        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
                
                # Create credentials from the parsed and fixed dictionary
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                st.success("Google Cloud credentials loaded successfully. Voice features are available.")
                
            except json.JSONDecodeError as json_err:
                st.error(f"Invalid JSON format in google_credentials: {str(json_err)}")
                # Log the first few characters for debugging (avoid logging entire credential)
                if len(creds_raw) > 20:
                    print(f"First 20 chars of credentials: {creds_raw[:20]}...")
                print(f"JSON parse error: {str(json_err)}")
        else:
            # If it's already a dictionary structure
            credentials = service_account.Credentials.from_service_account_info(creds_raw)
            st.success("Google Cloud credentials loaded successfully. Voice features are available.")
            
    except Exception as e:
        st.warning(f"Error loading Google Cloud credentials: {type(e).__name__}")
        # Print error details for debugging in logs
        print(f"Credential error type: {type(e).__name__}")
        print(f"Credential error details: {str(e)}")

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

    # Attach the markdown transcript
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
    # Check if credentials are available
    if credentials is None:
        st.warning("Speech-to-text unavailable. Please type your response instead.")
        return "Voice transcription unavailable. Please type your response."
    
    # Save audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        # Create speech client
        client = speech.SpeechClient(credentials=credentials)
        
        # Load the audio file
        with open(tmp_file_path, "rb") as audio_file:
            content = audio_file.read()
        
        # Configure the speech recognition request
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,  # Adjust if needed based on recording parameters
            language_code="en-GB",
        )
        
        # Perform speech recognition
        response = client.recognize(config=config, audio=audio)
        
        # Extract and return the transcript
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        return transcript
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return f"Error: {str(e)}"
    finally:
        # Delete the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def text_to_speech(text):
    # Check if credentials are available
    if credentials is None:
        return None
    
    try:
        # Create TTS client
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        
        # Set up the input text
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure the voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB",
            name="en-GB-Neural2-B",  # A neutral British voice
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        # Configure the audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Generate the speech
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Return the audio content
        return response.audio_content
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

def main():
    # --- Password authentication ---
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

    # --- Main interview content ---
    st.title("Rugby Taster Session Voice Interview Bot")

    # Check if Google credentials are available and show a message
    if credentials is None:
        st.warning("Google Cloud Speech services are not configured. Voice features will not be available.")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = (
            "Thank you for agreeing to speak with us about your recent rugby taster session. "
            "To begin, can you tell me a bit about yourself and any previous experience with rugby or other sports?"
        )
        # Convert initial question to speech
        if credentials is not None:
            try:
                st.session_state.current_audio = text_to_speech(st.session_state.current_question)
            except Exception as e:
                st.warning(f"Unable to generate speech: {e}")
                st.session_state.current_audio = None

    st.write("""
    **Information Sheet and Consent**  
    By ticking yes below, you consent to participate in this interview about your experience in a rugby taster session. 
    Your responses may be anonymously quoted in publications. You may end the interview at any time and request 
    your data be removed by emailing tony.myers@staff.newman.ac.uk. 
    An AI assistant will ask main questions and follow-up probing questions.
    """)

    # Consent checkbox
    consent = st.checkbox("I have read the information sheet and give my consent to participate in this interview.")

    if consent:
        # Display current question
        st.markdown(f"**AI Question:** {st.session_state.current_question}")
        
        # Play audio of current question if available
        if "current_audio" in st.session_state and st.session_state.current_audio:
            st.audio(st.session_state.current_audio, format="audio/mp3")
        
        # Voice recording section (only if Google credentials are available)
        if credentials is not None:
            st.write("**Speak your answer:**")
            audio_bytes = audio_recorder()
            
            # Process recorded audio
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
            st.info("Voice recording is not available. Please type your response below.")
            st.session_state.current_transcript = ""
        
        # Text area for editing transcription or typing response
        user_answer = st.text_area(
            "Your response (edit transcription or type):", 
            value=st.session_state.get("current_transcript", ""),
            key=f"user_input_{len(st.session_state.conversation)}"
        )

        # Radio buttons for rating 1–10
        user_rating = st.radio(
            "On a scale of 1–10, how would you rate your experience relating to the current question/topic?",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=4  # Pre-select '5'
        )

        # Progress
        completed_questions = len([entry for entry in st.session_state.conversation if entry['role'] == "user"])
        progress_percentage = min(completed_questions / total_questions, 1.0)
        st.write(f"**Interview Progress: {completed_questions} out of {total_questions} questions answered**")
        st.progress(progress_percentage)

        # Submit answer
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
                
                # Generate speech for the AI response if credentials are available
                if credentials is not None:
                    try:
                        st.session_state.current_audio = text_to_speech(ai_response)
                    except Exception as e:
                        st.warning(f"Unable to generate speech: {e}")
                        st.session_state.current_audio = None
                
                # Clear the current transcript
                st.session_state.current_transcript = ""
                
                st.experimental_rerun()
            else:
                st.warning("Please provide an answer before submitting.")

        # End Interview
        if st.button("End Interview"):
            st.success("Interview completed! Thank you for sharing your rugby taster session experience.")
            st.session_state.current_question = "Interview ended"

            # Convert conversation to markdown & send email
            transcript_md = convert_to_markdown(st.session_state.conversation)
            if send_email(transcript_md):
                st.info("Your transcript has been emailed to the researcher.")
            
            # Provide download link
            st.markdown(get_transcript_download_link(st.session_state.conversation), unsafe_allow_html=True)

        # Option to display transcript
        if st.checkbox("Show Interview Transcript"):
            st.write("**Interview Transcript:**")
            for entry in st.session_state.conversation:
                st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")
                st.write("---")

        # Restart Interview
        if st.button("Restart Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

if __name__ == "__main__":
    main()
