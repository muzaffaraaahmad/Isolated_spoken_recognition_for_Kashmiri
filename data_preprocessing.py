import os
import numpy as np
import librosa
from tqdm import tqdm

# Define the sample rate (fs)
fs = 16000  # Sample rate in Hz
frame_size = int(0.025 * fs)  # 25 ms frame size
overlap = int(0.020 * fs)     # 20 ms overlap size

# Function to normalize the mel-spectrogram
def normalize_mel_spectrogram(mel_spec):
    return (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)

# Function to extract features from audio files
def extract_features(dataset_path, fixed_size=32):
    feature_list = []
    label_list = []
    
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        
        for file in tqdm(os.listdir(class_path)):
            if not file.endswith('.wav'):
                continue
            
            audio_path = os.path.join(class_path, file)
            audio, sr = librosa.load(audio_path, sr=None)

            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size, hop_length=overlap, n_mels=42)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            normalized_mel_spec = normalize_mel_spectrogram(mel_spec_db)

            # Pad or truncate to a fixed size
            if normalized_mel_spec.shape[1] < fixed_size:
                pad_width = fixed_size - normalized_mel_spec.shape[1]
                padded_mel_spec = np.pad(normalized_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
            else:
                padded_mel_spec = normalized_mel_spec[:, :fixed_size]

            feature_list.append(padded_mel_spec.T)
            label_list.append(class_folder)

    return np.array(feature_list), np.array(label_list)
