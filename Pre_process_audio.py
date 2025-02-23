import os
import librosa
import soundfile as sf
import numpy as np
import random
from pydub import AudioSegment
from tqdm import tqdm

# Define paths
DATA_DIR = "your_data_directory"  # Change this to your dataset's root folder
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Categories and corresponding folders
CATEGORIES = {
    "crying": ["donateacry_corpus", "infant_cry"],
    "screaming": ["screaming"],
    "normal": ["utterances"]
}

# Parameters
TARGET_SR = 16000  # Standard sampling rate for YAMNet and Wav2Vec2
SEGMENT_DURATION = 5  # seconds
AUGMENTATION_PROB = 0.5  # 50% chance of augmentation

# Function to apply data augmentation
def augment_audio(y, sr):
    if random.random() < 0.5:
        # Add Gaussian noise
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise
    else:
        # Pitch shift randomly between -2 and +2 semitones
        y = librosa.effects.pitch_shift(y, sr, n_steps=random.uniform(-2, 2))
    return y

# Function to process audio files
def process_audio(file_path, label, output_folder):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Segment long audio into 5-sec chunks
        num_segments = max(1, int(duration / SEGMENT_DURATION))
        
        for i in range(num_segments):
            start_sample = i * SEGMENT_DURATION * sr
            end_sample = min((i + 1) * SEGMENT_DURATION * sr, len(y))
            segment = y[start_sample:end_sample]
            
            # Pad if too short
            if len(segment) < SEGMENT_DURATION * sr:
                pad_length = SEGMENT_DURATION * sr - len(segment)
                segment = np.pad(segment, (0, pad_length))
            
            # Data augmentation
            if random.random() < AUGMENTATION_PROB:
                segment = augment_audio(segment, sr)

            # Save processed file
            output_path = os.path.join(output_folder, f"{label}_{os.path.basename(file_path).replace('.mp3', '').replace('.wav', '')}_{i}.wav")
            sf.write(output_path, segment, sr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all files
for label, folders in CATEGORIES.items():
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        for subdir, _, files in os.walk(folder_path):
            for file in tqdm(files, desc=f"Processing {label} data"):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    file_path = os.path.join(subdir, file)
                    process_audio(file_path, label, OUTPUT_DIR)

print("Preprocessing complete!")
