import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

# --- CONFIGURATION ---
# Point this to your unzipped folder
MODEL_PATH = "./final_model"


class Transcriber:
    def __init__(self, model_path=MODEL_PATH):
        print(f"⏳ Loading Wav2Vec2 Model from {model_path}...")
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Folder {model_path} does not exist. Did you unzip the model?")

            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
            self.model.eval()  # Set to evaluation mode
            print("✅ Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.processor = None

    def transcribe(self, audio_path):
        if self.model is None:
            return "Error: Model not loaded."

        try:
            # 1. Load Audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # 2. Resample to 16kHz (Required by Wav2Vec2)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            # 3. Handle Stereo (Convert to Mono)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 4. Prepare Input
            input_values = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values

            # 5. Predict
            with torch.no_grad():
                logits = self.model(input_values).logits

            # 6. Decode (Numbers -> Text)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription.lower()

        except Exception as e:
            return f"Error processing file: {e}"


# Helper function for app.py
def load_transcriber():
    return Transcriber()