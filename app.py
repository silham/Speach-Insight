import streamlit as st
import os
import shutil
import time
import pandas as pd
from segmentation import segment_and_save
from model import load_transcriber

# --- UI CONFIGURATION ---
st.set_page_config(page_title="SpeechInSight | Pipeline", page_icon="🎙️", layout="wide")
st.title("🎙️ SpeechInSight: Monitoring Dashboard")
st.markdown("### 1. Upload Meeting Audio")


# --- 1. LOAD MODEL (Cached) ---
@st.cache_resource
def get_model():
    return load_transcriber()


# Initialize Transcriber
transcriber = get_model()

if transcriber.model is None:
    st.error("❌ Model failed to load. Please check terminal for errors.")
    st.stop()

# --- 2. UPLOAD SECTION ---
uploaded_file = st.file_uploader("Drop audio here (WAV, MP3)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Save input temporarily
    input_path = "temp_input.wav"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 2])

    with col1:
        st.success("File Uploaded!")
        st.audio(input_path)

        if st.button("🚀 Start Analysis Pipeline"):
            output_folder = "pipeline_output"
            progress_bar = st.progress(0)
            status_text = st.empty()

            # --- STEP 1: SEGMENTATION (DIARIZATION) ---
            status_text.text("✂️ Step 1: identifying speakers & segmenting...")
            clips = segment_and_save(input_path, output_folder)

            if not clips:
                st.error("❌ No segments found! Check Pyannote token or audio quality.")
                st.stop()

            progress_bar.progress(30)

            # --- STEP 2: TRANSCRIPTION ---
            status_text.text("🧠 Step 2: Transcribing speech...")
            results = []

            for i, clip_path in enumerate(clips):
                # Transcribe
                text = transcriber.transcribe(clip_path)

                # Parse filename to get speaker (seg_001_SPEAKER_00.wav)
                filename = os.path.basename(clip_path)
                parts = filename.split('_')
                speaker_id = f"{parts[2]}_{parts[3].split('.')[0]}" if len(parts) >= 4 else "Unknown"

                results.append({
                    "Speaker": speaker_id,
                    "Transcript": text,
                    "File": filename,
                    "Path": clip_path
                })

                # Update progress
                current_prog = 30 + int((i / len(clips)) * 70)
                progress_bar.progress(current_prog)

            progress_bar.progress(100)
            status_text.success("✅ Analysis Complete!")

            # --- RESULTS ---
            with col2:
                st.subheader("📊 Conversation Log")
                df = pd.DataFrame(results)

                # Display colorful table
                st.dataframe(df[["Speaker", "Transcript"]], use_container_width=True)

                # Export Options
                st.divider()
                st.subheader("📥 Export Data")

                # 1. CSV for You
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Download Transcripts (CSV)",
                    csv,
                    "meeting_transcripts.csv",
                    "text/csv"
                )

                # 2. ZIP for Teammate
                shutil.make_archive("audio_segments", 'zip', output_folder)
                with open("audio_segments.zip", "rb") as fp:
                    st.download_button(
                        "📦 Download Audio Clips (ZIP)",
                        fp,
                        "audio_segments.zip",
                        "application/zip"
                    )