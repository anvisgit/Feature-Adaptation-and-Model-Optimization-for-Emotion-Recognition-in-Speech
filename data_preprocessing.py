import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from config import *
from features import extractFeatures, extractStatisticalFunctionals, extractMelSpec2D, extractMultiFeature

logger = logging.getLogger(__name__)

def loadDataset(path):
    data = []
    logger.info(f"Loading dataset from {path}")
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(subdir, file)
                part = file.split('.')[0].split('-')
                if len(part) < 3:
                    continue
                emotionCode = part[2]
                actorId = part[-1]
                if emotionCode in EMOTION_TO_ID:
                    data.append({
                        'path': filepath,
                        'emotion': ID_TO_EMOTION[EMOTION_TO_ID[emotionCode]],
                        'actor': int(actorId),
                        'emotionId': EMOTION_TO_ID[emotionCode]
                    })
    df = pd.DataFrame(data)
    logger.info(f"Loaded DataFrame with {len(df)} samples")
    return df

def augmentAudio(audio, sr):
    augmented = audio.copy()

    if np.random.rand() < 0.6:
        noise = np.random.randn(len(augmented))
        augmented = augmented + AUGMENTATION['noiseFactor'] * noise

    if np.random.rand() < 0.6:
        nSteps = np.random.uniform(-1.5, 1.5)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=nSteps)

    if np.random.rand() < 0.6:
        speed = np.random.uniform(0.9, 1.1)
        augmented = librosa.effects.time_stretch(augmented, rate=speed)

    if np.random.rand() < 0.4:
        vtlpFactor = np.random.uniform(0.9, 1.1)
        stft = librosa.stft(augmented, n_fft=N_FFT, hop_length=HOP_LENGTH)
        freqs = np.linspace(0, 1, stft.shape[0])
        warpedFreqs = freqs ** vtlpFactor
        warpedFreqs = np.clip(warpedFreqs * (stft.shape[0] - 1), 0, stft.shape[0] - 1).astype(int)
        warpedStft = stft[warpedFreqs, :]
        augmented = librosa.istft(warpedStft, hop_length=HOP_LENGTH, length=len(augmented))

    return augmented

def loadDataMulti(path, emotions, nmfcc=40, nmels=128, nfft=2048, hop=512, ntecc=40):
    xMfcc, xMel, xChroma, xTecc, y, audios, actorIds = [], [], [], [], [], [], []
    logger.info(f"Loading multi-feature data: MFCC({nmfcc}+d) + Mel({nmels}) + Chroma(12+d) + TECC({ntecc}+d)")
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
                        mfcc, mel, chroma, tecc = extractMultiFeature(audio, sr, nmfcc, nmels, nfft, hop, ntecc)
                        xMfcc.append(mfcc)
                        xMel.append(mel)
                        xChroma.append(chroma)
                        xTecc.append(tecc)
                        y.append(emotions[p[2]])
                        audios.append(audio)
                        actorIds.append(int(p[-1]))
                    except Exception as e:
                        logger.error(f"Error processing {filepath}: {e}")
    
    return [np.array(xMfcc, dtype=np.float32), 
            np.array(xMel, dtype=np.float32), 
            np.array(xChroma, dtype=np.float32),
            np.array(xTecc, dtype=np.float32)], \
           np.array(y), audios, np.array(actorIds)

def applySpecAugment(feat, numFreqMasks=2, numTimeMasks=2, freqMaskParam=None, timeMaskParam=10):
    augFeat = feat.copy()
    T, F = augFeat.shape
    if freqMaskParam is None:
        freqMaskParam = max(1, F // 5)
    for _ in range(numFreqMasks):
        fWidth = np.random.randint(1, min(freqMaskParam, F))
        f0 = np.random.randint(0, max(1, F - fWidth))
        augFeat[:, f0:f0 + fWidth] = 0
    for _ in range(numTimeMasks):
        tWidth = np.random.randint(1, min(timeMaskParam, T))
        t0 = np.random.randint(0, max(1, T - tWidth))
        augFeat[t0:t0 + tWidth, :] = 0
    return augFeat

def augmentAudioBatchMulti(audios, labels, nmfcc=40, nmels=128, nfft=2048, hop=512, ntecc=40, numAug=4):
    mfccAug, melAug, chromaAug, teccAug, yAug = [], [], [], [], []
    targetLen = int(SAMPLE_RATE * AUDIO_DURATION)
    neutralId = 0
    for i, audio in enumerate(audios):
        augCount = numAug * 2 if labels[i] == neutralId else numAug
        for _ in range(augCount):
            try:
                augAudio = augmentAudio(audio, SAMPLE_RATE)
                if len(augAudio) < targetLen:
                    augAudio = np.pad(augAudio, (0, targetLen - len(augAudio)))
                else:
                    augAudio = augAudio[:targetLen]
                mfcc, mel, chroma, tecc = extractMultiFeature(augAudio, SAMPLE_RATE, nmfcc, nmels, nfft, hop, ntecc)
                if np.random.rand() < 0.5:
                    mfcc = applySpecAugment(mfcc)
                    mel = applySpecAugment(mel)
                    chroma = applySpecAugment(chroma, freqMaskParam=6)
                    tecc = applySpecAugment(tecc)
                mfccAug.append(mfcc)
                melAug.append(mel)
                chromaAug.append(chroma)
                teccAug.append(tecc)
                yAug.append(labels[i])
            except Exception as e:
                logger.error(f"Augmentation error: {e}")
    return [np.array(mfccAug, dtype=np.float32), np.array(melAug, dtype=np.float32),
            np.array(chromaAug, dtype=np.float32), np.array(teccAug, dtype=np.float32)], np.array(yAug)

def loadData(path, emotions, featType='mfcc', **featParams):
    x, y, audios, actorIds = [], [], [], []
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
                        actorIds.append(int(p[-1]))
                        if featType == 'mel2d':
                            feat = extractMelSpec2D(audio, sr, **featParams)
                            x.append(feat)
                        else:
                            stat, delta, deltaDelta = extractFeatures(audio, sr, featType=featType, **featParams)
                            feat = np.hstack((stat, delta, deltaDelta))
                            x.append(feat)
                        y.append(emotions[p[2]])
                    except Exception as e:
                        logger.error(f"Error processing {filepath}: {e}")
    return np.array(x), np.array(y), audios, np.array(actorIds)
