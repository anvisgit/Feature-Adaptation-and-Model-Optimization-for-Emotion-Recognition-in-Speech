import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import config 
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def add_noise(data):
    noise_amp = config.NOISE_AMPLITUDE_FACTOR * np.amax(data) * np.random.uniform()
    augmented = data + np.random.normal(size=data.shape[0]) * noise_amp
    return augmented

def time_stretch(data, rate=None):
    if rate is None:
        rate = config.TIME_STRETCH_RATE
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sr, n_steps=None):
    if n_steps is None:
        n_steps = config.PITCH_SHIFT_STEPS
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def extract_features(data, sr):
    result = np.array([])
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    
    stft = np.abs(librosa.stft(data, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma))
    
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=config.N_MFCC).T, axis=0)
    result = np.hstack((result, mfcc))
    
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

def get_features(path):
    try:
        data, sr = librosa.load(path, duration=config.DURATION, offset=config.AUDIO_OFFSET)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return np.array([])
    
    res1 = extract_features(data, sr)
    feature_list = [res1]

    if config.AUGMENT_NOISE:
        noise_data = add_noise(data)
        res2 = extract_features(noise_data, sr)
        feature_list.append(res2)
        
    if config.AUGMENT_TIME_STRETCH and config.AUGMENT_PITCH:
        new_data = pitch_shift(time_stretch(data), sr)
        res3 = extract_features(new_data, sr)
        feature_list.append(res3)
    
    return np.array(feature_list)

def load_data(data_dir=config.DATA_PATH):
    x, y = [], []
    file_pattern = os.path.join(data_dir, "**/*.wav")
    files = glob.glob(file_pattern, recursive=True)
    if not files:
        logger.error(f"No files found in {data_dir}. Please check the path in config.py")
        return np.array([]), np.array([])

    logger.info(f"Found {len(files)} files. Starting feature extraction.")
    
    missing_emotions = 0
    processed_count = 0
    
    for file in tqdm(files, desc="Extracting features"):
        try:
            file_name = os.path.basename(file)
            parts = file_name.split("-")
            
            if len(parts) < 3:
                logger.debug(f"Skipping malformed filename: {file_name}")
                continue
                
            emotion_code = parts[2]
            if emotion_code not in config.OBSERVED_EMOTIONS:
                missing_emotions += 1
                continue
                
            emotion = config.OBSERVED_EMOTIONS[emotion_code]
            processed_count += 1
            
            features = get_features(file)
            
            if features.size == 0:
                continue
            
            for feature in features:
                x.append(feature)
                y.append(emotion)
        except Exception as e:
            logger.warning(f"Error processing {file}: {e}")
            continue
            
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    if len(x) > 0:
        logger.info(f"Feature Extraction Complete. Total Samples (with aug): {len(x)}")
        logger.info(f"Feature Vector Shape: {x_arr.shape}")
        
        unique, counts = np.unique(y_arr, return_counts=True)
        dist = dict(zip(unique, counts))
        logger.info(f"Class Distribution: {dist}")
    else:
        logger.warning("No valid samples were extracted.")

    if missing_emotions > 0:
        logger.info(f"Skipped {missing_emotions} files due to unobserved emotion codes.")

    return x_arr, y_arr

if __name__ == "__main__":
    pass
