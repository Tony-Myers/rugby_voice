test_audio_bytes = audio_recorder(
                pause_threshold=2.0,  # Longer pause threshold before stopping
                recording_color="#FF5733",  # More visible recording color
                neutral_color="#6aa36f"
            )
            if test_audio_bytes:
                st.success("Audio recorded!")
                
                # Add download link for the audio file for debugging
                st.write("Download the recorded audio for troubleshooting:")
                st.markdown(get_audio_download_link(test_audio_bytes, "audio/wav", "test_recording.wav"), unsafe_allow_html=True)
                
                with st.spinner("Transcribing your speech..."):
                    transcript = transcribe_audio(test_audio_bytes)
                    if transcript and not transcript.startswith("Error:"):
                        st.success("Transcription successful!")
                        st.write(f"**Transcribed Text:** {transcript}")
                    else:
                        st.error("Transcription failed.")
                        st.write(f"Details: {transcript}")
        else:
            st.error("No credentials available for voice recognition test.")

    # --- Detailed TTS Test button ---
    st.write("### Detailed TTS Test")
    st.write("This test will provide more information to diagnose TTS issues.")
    if st.button("Run Detailed TTS Test"):
        if credentials:
            with st.spinner("Testing TTS with detailed diagnostics..."):
                message, audio_bytes, mime_type = test_tts_detailed()
                st.write(message)
                if audio_bytes:
                    st.audio(audio_bytes, format=mime_type)
                    st.markdown(get_audio_download_link(audio_bytes, mime_type, f"detailed_test.{mime_type.split('/')[-1]}"), unsafe_allow_html=True)
                    st.success(f"Successfully generated {len(audio_bytes)} bytes of audio. If you can't hear it, try downloading it.")
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
                message, audio_bytes, mime_type = try_all_tts_voices()
                st.write(message)
                if audio_bytes:
                    st.audio(audio_bytes, format=mime_type)
                    st.markdown(get_audio_download_link(audio_bytes, mime_type, f"voice_test.{mime_type.split('/')[-1]}"), unsafe_allow_html=True)
                    st.success(f"Successfully generated {len(audio_bytes)} bytes of audio. If you can't hear it, try downloading it.")
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
                audio_bytes, mime_type = text_to_speech(st.session_state.current_question)
                info_str = debug_print_tts(audio_bytes, label="Initial Q TTS")
                if audio_bytes and len(audio_bytes) > 0:
                    st.session_state.current_audio = audio_bytes
                    st.session_state.current_audio_mime = mime_type
                    st.success("Initial question audio generated successfully.")
                else:
                    st.warning(f"No audio returned for the initial question. Debug: {info_str}")
                    st.session_state.current_audio = None
                    st.session_state.current_audio_mime = None
            except Exception as e:
                st.warning(f"Unable to generate speech for initial question: {e}")
                st.session_state.current_audio = None
                st.session_state.current_audio_mime = None

    # Initialize session state for recording management
    if "current_recordings" not in st.session_state:
        st.session_state.current_recordings = []
    if "show_recordings_ui" not in st.session_state:
        st.session_state.show_recordings_ui = False
    
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
        
        # If we have TTS audio, show it with autoplay option
        if "current_audio" in st.session_state and st.session_state.current_audio:
            mime_type = st.session_state.get("current_audio_mime", "audio/wav")
            st.caption(f"(Audio question will play automatically. Length = {len(st.session_state.current_audio)} bytes)")
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Try to use autoplay
                st.markdown(get_autoplay_audio_html(st.session_state.current_audio, mime_type), unsafe_allow_html=True)
                
                # Also provide standard audio player as fallback
                st.audio(st.session_state.current_audio, format=mime_type)
            
            with col2:
                # Download link
                st.markdown(get_audio_download_link(st.session_state.current_audio, mime_type, "question_audio.wav"), unsafe_allow_html=True)
                
                # Explicit play button (more obvious)
                if st.button("🔊 Play Question"):
                    # This forces a rerun which helps with autoplay
                    st.rerun()
            
            if st.button("Regenerate Question Audio"):
                try:
                    audio_bytes, mime_type = text_to_speech(st.session_state.current_question)
                    dbg = debug_print_tts(audio_bytes, label="Regenerated Q TTS")
                    if audio_bytes and len(audio_bytes) > 0:
                        st.session_state.current_audio = audio_bytes
                        st.session_state.current_audio_mime = mime_type
                        st.success(f"Audio regenerated successfully. {dbg}")
                    else:
                        st.warning(f"No audio returned. {dbg}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error regenerating audio: {e}")
        else:
            st.warning("Audio for this question is not available. You can try generating it.")
            if st.button("Generate Audio for Question"):
                try:
                    audio_bytes, mime_type = text_to_speech(st.session_state.current_question)
                    dbg = debug_print_tts(audio_bytes, label="On-demand Q TTS")
                    if audio_bytes and len(audio_bytes) > 0:
                        st.session_state.current_audio = audio_bytes
                        st.session_state.current_audio_mime = mime_type
                        st.success(f"Audio generated successfully. {dbg}")
                        st.rerun()
                    else:
                        st.error(f"Failed to generate audio (0 bytes). {dbg}")
                except Exception as e:
                    st.error(f"Error generating audio: {e}")
        
        # Improved voice recording section with concatenation support
        if credentials is not None:
            st.write("---")
            st.write("**Speak your answer:**")
            st.write("Continue speaking until you've completed your answer. If recording stops, you can click record again to continue.")
            
            # Display recording count
            if st.session_state.current_recordings:
                st.write(f"**{len(st.session_state.current_recordings)} recording(s) captured so far**")
            
            # Record audio with better parameters
            audio_bytes = audio_recorder(
                pause_threshold=2.0,  # Longer pause threshold before stopping
                recording_color="#FF5733",  # More visible recording color
                neutral_color="#6aa36f",
                energy_threshold=0.01,  # Lower threshold to detect quieter speech
            )
            
            if audio_bytes:
                # Store the new recording
                st.session_state.current_recordings.append(audio_bytes)
                st.success(f"Recording {len(st.session_state.current_recordings)} captured! You can continue recording if needed.")
                st.session_state.show_recordings_ui = True
                st.rerun()  # Refresh to show the updated UI
            
            # Show UI for managing recordings
            if st.session_state.show_recordings_ui and st.session_state.current_recordings:
                if st.button("Process All Recordings"):
                    with st.spinner("Combining recordings and transcribing..."):
                        # Combine all recordings
                        combined_audio = combine_audio_segments(st.session_state.current_recordings)
                        
                        # Show download link for debugging
                        st.write("Download the combined recording for troubleshooting:")
                        st.markdown(get_audio_download_link(combined_audio, "audio/wav", "combined_recording.wav"), unsafe_allow_html=True)
                        
                        # Transcribe the combined audio
                        transcript = transcribe_audio(combined_audio)
                        st.session_state.current_transcript = transcript
                        st.write(f"**Transcribed:** {transcript}")
                
                # Clear recordings button
                if st.button("Clear recordings and start over"):
                    st.session_state.current_recordings = []
                    st.session_state.current_transcript = ""
                    st.session_state.show_recordings_ui = False
                    st.rerun()
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
                        next_audio, mime_type = text_to_speech(ai_response)
                        dbg = debug_print_tts(next_audio, label="Follow-up TTS")
                        if next_audio and len(next_audio) > 0:
                            st.session_state.current_audio = next_audio
                            st.session_state.current_audio_mime = mime_type
                        else:
                            st.warning(f"Failed to generate audio for follow-up. {dbg}")
                            st.session_state.current_audio = None
                            st.session_state.current_audio_mime = None
                    except Exception as e:
                        st.warning(f"Unable to generate speech: {e}")
                        st.session_state.current_audio = None
                        st.session_state.current_audio_mime = None
                
                # Reset recording state for next question
                st.session_state.current_transcript = ""
                st.session_state.current_recordings = []
                st.session_state.show_recordings_ui = False
                
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
