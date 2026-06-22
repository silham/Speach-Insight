# Semantic Segmentation Guide: Splitting Long Monologues by Topic

This guide explains how to implement **Semantic Splitting (Method 1)** in the `SpeechInSight` pipeline. This method uses your existing `TemplateClassifier` to detect topic shifts (e.g., from "WarmUp" to "Praise") and splits long speaker turns into smaller, logically separated segments. 

It also includes the **Approximation Hack** to split the `.wav` audio files based on character length so that the frontend can play the audio for each sub-segment.

---

## Overview of the Implementation

We will inject this logic into `pipeline/__init__.py`, right after the transcription stage but before the emotion analysis stage. 

The flow will be:
1. **Transcribe** the Pyannote segment (as usual).
2. **Tokenize** the transcript into sentences using `nltk`.
3. **Classify** each sentence and detect boundaries where the label changes.
4. **Calculate** the audio split points using character ratios (The Approximation Hack).
5. **Split** the audio file using `librosa` or `soundfile` and save the new `.wav` files.
6. **Replace** the old, long `SegmentResult` with the new, shorter `SegmentResult` objects in the pipeline's job state.

---

## Step 1: Install NLTK for Sentence Splitting

You will need a robust way to split text into sentences. `nltk` is standard and lightweight for this.

Run this in your terminal to install NLTK:
```bash
pip install nltk
```

Then, you'll need to download the `punkt` tokenizer model once. You can add this small script to run once, or put it at the top of your `pipeline/__init__.py`:
```python
import nltk
nltk.download('punkt')
```

---

## Step 2: Write the Audio Slicing Utility

We need a quick function to slice a `.wav` file given a start and end time. You can add this to your `media_utils.py` file. Since you already use `soundfile` and `librosa`, we can use them to cut the audio.

```python
# In media_utils.py
import soundfile as sf
import librosa
import os

def slice_audio(input_path: str, output_path: str, start_ratio: float, end_ratio: float):
    """
    Slices a WAV file based on a percentage of its total duration.
    start_ratio and end_ratio should be between 0.0 and 1.0.
    """
    # Load the audio array and sample rate
    y, sr = librosa.load(input_path, sr=16000)
    total_samples = len(y)
    
    # Calculate sample indices based on the ratios
    start_sample = int(total_samples * start_ratio)
    end_sample = int(total_samples * end_ratio)
    
    # Slice the array
    y_slice = y[start_sample:end_sample]
    
    # Save the new sliced audio
    sf.write(output_path, y_slice, sr)
```

---

## Step 3: Implement the Logic in the Pipeline

Now we modify `pipeline/__init__.py`. We will create a helper method to handle the splitting, and then call it during the main pipeline run.

Add this method to your `AnalysisPipeline` class in `pipeline/__init__.py`:

```python
import os
from nltk.tokenize import sent_tokenize
from .schemas import SegmentResult, SegmentMeta
from media_utils import slice_audio

class AnalysisPipeline:
    # ... your existing __init__ ...

    def _split_segment_semantically(self, seg: SegmentResult, job_id: str, job_output_folder: str) -> list[SegmentResult]:
        """
        Splits a single long SegmentResult into multiple SegmentResults 
        based on template classification boundary detection.
        """
        # 1. Split transcript into sentences
        sentences = sent_tokenize(seg.text)
        if len(sentences) <= 1:
            return [seg] # Nothing to split

        sub_segments = []
        current_label = None
        current_sentences = []
        
        # We need to track character lengths for the audio approximation hack
        total_chars = len(seg.text)
        current_start_char = 0
        
        # 2. Iterate through sentences and detect boundaries
        for sentence in sentences:
            # Classify the single sentence
            result = self.template_classifier.classify(sentence)
            label = result.get("label", "neutral")
            
            # If label changes (and it's not the very first sentence), we hit a boundary!
            if label != current_label and current_label is not None:
                # Calculate character boundaries for the group we just finished
                group_text = " ".join(current_sentences)
                group_char_len = len(group_text)
                
                start_ratio = current_start_char / total_chars
                end_ratio = (current_start_char + group_char_len) / total_chars
                
                # Create a new audio file for this sub-segment
                new_audio_name = f"{os.path.splitext(os.path.basename(seg.audio_path))[0]}_sub_{len(sub_segments)}.wav"
                new_audio_path = os.path.join(job_output_folder, new_audio_name)
                
                # SLICE THE AUDIO (The Approximation Hack)
                slice_audio(seg.audio_path, new_audio_path, start_ratio, end_ratio)
                
                # Create the new SegmentResult
                audio_url = f"{self.audio_base_url}/{job_id}/{new_audio_name}"
                new_seg = SegmentResult.from_meta(SegmentMeta(
                    segment_id=seg.segment_id, # We will re-index these later
                    speaker=seg.speaker,
                    start_time=seg.start_time + ((seg.end_time - seg.start_time) * start_ratio),
                    end_time=seg.start_time + ((seg.end_time - seg.start_time) * end_ratio),
                    audio_path=new_audio_path,
                    audio_url=audio_url
                ))
                new_seg.text = group_text
                # We can safely pre-assign the template label here!
                new_seg.template_label = current_label 
                sub_segments.append(new_seg)
                
                # Advance the character pointer
                current_start_char += group_char_len + 1 # +1 for the space
                
                # Reset for the new group
                current_sentences = []

            current_sentences.append(sentence)
            current_label = label

        # 3. Don't forget the final group!
        if current_sentences:
            group_text = " ".join(current_sentences)
            start_ratio = current_start_char / total_chars
            end_ratio = 1.0
            
            new_audio_name = f"{os.path.splitext(os.path.basename(seg.audio_path))[0]}_sub_{len(sub_segments)}.wav"
            new_audio_path = os.path.join(job_output_folder, new_audio_name)
            
            slice_audio(seg.audio_path, new_audio_path, start_ratio, end_ratio)
            
            audio_url = f"{self.audio_base_url}/{job_id}/{new_audio_name}"
            new_seg = SegmentResult.from_meta(SegmentMeta(
                segment_id=seg.segment_id,
                speaker=seg.speaker,
                start_time=seg.start_time + ((seg.end_time - seg.start_time) * start_ratio),
                end_time=seg.end_time,
                audio_path=new_audio_path,
                audio_url=audio_url
            ))
            new_seg.text = group_text
            new_seg.template_label = current_label
            sub_segments.append(new_seg)

        return sub_segments
```

---

## Step 4: Inject into the Main Pipeline Run

Finally, update the `run()` method in `pipeline/__init__.py` to use your new splitting logic. You should do this **after transcription**, but **before emotion analysis**.

```python
        # ── Stage 2: Transcription ──────────────────────────────────────
        print(f"🧠 Transcribing {len(job.segments)} segments…")
        for seg in job.segments:
            seg.text = self.transcriber.transcribe(seg.audio_path)


        # ── NEW STAGE: Semantic Splitting ───────────────────────────────
        if self.template_classifier is not None:
            print(f"✂️  Splitting long segments semantically...")
            new_segments_list = []
            for seg in job.segments:
                # If segment is longer than e.g., 20 seconds, we attempt to split it
                if (seg.end_time - seg.start_time) > 20.0:
                    split_segs = self._split_segment_semantically(seg, job_id, job_output_folder)
                    new_segments_list.extend(split_segs)
                else:
                    new_segments_list.append(seg)
                    
            # Update the job segments with our newly split segments
            job.segments = new_segments_list
            
            # Re-index the segment IDs so they remain sequential
            for i, seg in enumerate(job.segments):
                seg.segment_id = i


        # ── Stage 3: Emotion Recognition ────────────────────────────────
        print(f"🎭 Running emotion analysis on {len(job.segments)} segments…")
        # ... your existing emotion analysis code ...
```

## Summary of Changes
1. Added NLTK for robust sentence splitting.
2. Created a quick `slice_audio` utility using maths to estimate where in the audio file a text topic starts and ends.
3. Added the splitting loop which checks for template classification boundaries.
4. Rewrote the `job.segments` array so the rest of your pipeline (Emotion, Scoring) treats these new sub-segments exactly like regular Pyannote segments.

*Note on Stage 4 (Template Classification):* Since we already classified the sentences and grouped them, we technically assigned the `template_label` during the split. You can either leave your existing Stage 4 alone (it will just re-classify the newly grouped text block, which is fine), or optimize it to skip segments that already have a `template_label`!
