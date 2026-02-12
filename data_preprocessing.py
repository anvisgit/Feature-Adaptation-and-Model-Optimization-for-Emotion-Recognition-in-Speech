import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from config import *
from features import extractFeatures, extractStatisticalFunctionals

logger = logging.getLogger(__name__)

def loadDataset(path):
    data = []
    logger.info(f"Loading dataset metadata from {path}")
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(subdir, file)
                part = file.split('.')[0].split('-')
                if len(part) < 3: continue
                emotionCode = part[2]
                actorId = part[-1]
                if emotionCode in EMOTION_TO_ID:
                    data.append({
                        'path': filepath,
                        'emotion': ID_TO_EMOTION[EMOTION_TO_ID[emotionCode]],
                        'actor': int(actorId),
                        'emotion_id': EMOTION_TO_ID[emotionCode]
                    })
    df = pd.DataFrame(data)
    logger.info(f"Loaded DataFrame with {len(df)} samples")
    return df

def augmentAudio(audio, sr):
    augmented = audio.copy()
    if np.random.rand() < 0.5:
        noise = np.random.randn(len(augmented))
        augmented = augmented + AUGMENTATION['noise_factor'] * noise
    if np.random.rand() < 0.5:
        nSteps = np.random.uniform(-AUGMENTATION['pitch_steps'], AUGMENTATION['pitch_steps'])
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=nSteps)
    if np.random.rand() < 0.5:
        speed = np.random.uniform(AUGMENTATION['speed_range'][0], AUGMENTATION['speed_range'][1])
        augmented = librosa.effects.time_stretch(augmented, rate=speed)
    return augmented

def loadData(path, emotions, featType='mfcc', **featParams):
    x, y, audios = [], [], []
    logger.info(f"Loading data: {featType}")
    targetLen = int(SAMPLE_RATE * AUDIO_DURATION)
    for subdir, dirs, fs in os.walk(path):
        for f in fs:
            if f.endswith('.wav'):
                p = f.split('.')[0].split('-')
                if len(p) >= 3 and p[2] in emotions:
                    filepath = os.path.join(subdir, f)
                    try:
                        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=AUDIO_DURATION)
                        audio = librosa.effects.trim(audio)[0]
                        if len(audio) < targetLen:
                            audio = np.pad(audio, (0, targetLen - len(audio)))
                        else:
                            audio = audio[:targetLen]
                        audios.append(audio)
                        stat, delta, deltaDelta = extractFeatures(audio, sr, featType=featType, **featParams)
                        feat = np.hstack((stat, delta, deltaDelta))
                        x.append(feat)
                        y.append(emotions[p[2]])
                    except Exception as e:
                        logger.error(f"Error {filepath}: {e}")
    xArr = np.array(x)
    yArr = np.array(y)
    logger.info(f"Loaded: X={xArr.shape}, y={yArr.shape}")
    return xArr, yArr, audios

def augmentAudioBatch(audios, labels, featType, featParams, numAug=4):
    logger.info(f"Audio-domain augmentation x{numAug}")
    xAug, yAug = [], []
    targetLen = int(SAMPLE_RATE * AUDIO_DURATION)
    for i, audio in enumerate(audios):
        for _ in range(numAug):
            try:
                augAudio = augmentAudio(audio, SAMPLE_RATE)
                if len(augAudio) < targetLen:
                    augAudio = np.pad(augAudio, (0, targetLen - len(augAudio)))
                else:
                    augAudio = augAudio[:targetLen]
                stat, delta, deltaDelta = extractFeatures(augAudio, SAMPLE_RATE, featType=featType, **featParams)
                feat = np.hstack((stat, delta, deltaDelta))
                xAug.append(feat)
                yAug.append(labels[i])
            except Exception as e:
                logger.error(f"Augmentation error sample {i}: {e}")
    return np.array(xAug), np.array(yAug)

def preprocessForMlp(x3d):
    logger.info("Extracting statistical functionals for MLP")
    xFlat = extractStatisticalFunctionals(x3d)
    scaler = StandardScaler()
    xScaled = scaler.fit_transform(xFlat)
    logger.info(f"MLP feature shape: {xScaled.shape}")
    return xScaled, scaler

def encodeLabels(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y), encoder
