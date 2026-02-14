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
    logger.debug("Applying audio augmentation")
    augmented = audio.copy()
    applied = False

    if np.random.rand() < 0.7:
        logger.debug("Applying additive noise augmentation")
        noise = np.random.randn(len(augmented))
        augmented = augmented + AUGMENTATION['noiseFactor'] * noise
        applied = True

    if np.random.rand() < 0.7:
        nSteps = np.random.uniform(-AUGMENTATION['pitchSteps'], AUGMENTATION['pitchSteps'])
        logger.debug(f"Applying pitch shift by {nSteps:.2f} semitones")
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=nSteps)
        applied = True

    if np.random.rand() < 0.7:
        speed = np.random.uniform(AUGMENTATION['speedRange'][0], AUGMENTATION['speedRange'][1])
        logger.debug(f"Applying time stretch at rate {speed:.2f}")
        augmented = librosa.effects.time_stretch(augmented, rate=speed)
        applied = True

    if not applied:
        logger.debug("No augmentation randomly selected, forcing noise injection")
        noise = np.random.randn(len(augmented))
        augmented = augmented + AUGMENTATION['noiseFactor'] * noise

    return augmented

def loadDataMulti(path, emotions, nmfcc=40, nmels=128, nfft=2048, hop=512):
    xMfcc, xMel, xChroma, y, audios = [], [], [], [], []
    logger.info(f"Loading multi-feature data: MFCC({nmfcc}) + Mel({nmels}) + Chroma(12)")
    targetLen = int(SAMPLE_RATE * AUDIO_DURATION)
    logger.info(f"Target audio length {targetLen} samples at {SAMPLE_RATE}Hz for {AUDIO_DURATION}s")
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
                        mfcc, mel, chroma = extractMultiFeature(audio, sr, nmfcc, nmels, nfft, hop)
                        xMfcc.append(mfcc)
                        xMel.append(mel)
                        xChroma.append(chroma)
                        y.append(emotions[p[2]])
                        audios.append(audio)
                    except Exception as e:
                        logger.error(f"Error processing {filepath}: {e}")
    xMfccArr = np.array(xMfcc, dtype=np.float32)
    xMelArr = np.array(xMel, dtype=np.float32)
    xChromaArr = np.array(xChroma, dtype=np.float32)
    yArr = np.array(y)
    logger.info(f"Loaded multi-feature data: MFCC={xMfccArr.shape} Mel={xMelArr.shape} Chroma={xChromaArr.shape} y={yArr.shape}")
    return [xMfccArr, xMelArr, xChromaArr], yArr, audios

def augmentAudioBatchMulti(audios, labels, nmfcc=40, nmels=128, nfft=2048, hop=512, numAug=4):
    logger.info(f"Performing multi-feature audio augmentation x{numAug} for {len(audios)} samples")
    mfccAug, melAug, chromaAug, yAug = [], [], [], []
    targetLen = int(SAMPLE_RATE * AUDIO_DURATION)
    for i, audio in enumerate(audios):
        for _ in range(numAug):
            try:
                augAudio = augmentAudio(audio, SAMPLE_RATE)
                if len(augAudio) < targetLen:
                    augAudio = np.pad(augAudio, (0, targetLen - len(augAudio)))
                else:
                    augAudio = augAudio[:targetLen]
                mfcc, mel, chroma = extractMultiFeature(augAudio, SAMPLE_RATE, nmfcc, nmels, nfft, hop)
                mfccAug.append(mfcc)
                melAug.append(mel)
                chromaAug.append(chroma)
                yAug.append(labels[i])
            except Exception as e:
                logger.error(f"Augmentation error for sample {i}: {e}")
    logger.info(f"Multi-feature augmentation complete: generated {len(yAug)} augmented samples")
    return [np.array(mfccAug, dtype=np.float32), np.array(melAug, dtype=np.float32), np.array(chromaAug, dtype=np.float32)], np.array(yAug)

def loadData(path, emotions, featType='mfcc', **featParams):
    x, y, audios, actorIds = [], [], [], []
    logger.info(f"Loading data with feature type {featType}")
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
                        actorId = int(p[-1])
                        actorIds.append(actorId)
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
    xArr = np.array(x)
    yArr = np.array(y)
    actorIdsArr = np.array(actorIds)
    logger.info(f"Loaded data X={xArr.shape} y={yArr.shape} uniqueActors={len(np.unique(actorIdsArr))}")
    return xArr, yArr, audios, actorIdsArr

def augmentAudioBatch(audios, labels, featType, featParams, numAug=4):
    logger.info(f"Performing audio augmentation x{numAug} for {len(audios)} samples")
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
                if featType == 'mel2d':
                    feat = extractMelSpec2D(augAudio, SAMPLE_RATE, **featParams)
                    xAug.append(feat)
                else:
                    stat, delta, deltaDelta = extractFeatures(augAudio, SAMPLE_RATE, featType=featType, **featParams)
                    feat = np.hstack((stat, delta, deltaDelta))
                    xAug.append(feat)
                yAug.append(labels[i])
            except Exception as e:
                logger.error(f"Augmentation error for sample {i}: {e}")
    logger.info(f"Augmentation complete generated {len(xAug)} augmented samples")
    return np.array(xAug), np.array(yAug)

def preprocessForMlp(x3d):
    logger.info("Extracting statistical functionals for MLP input")
    xFlat = extractStatisticalFunctionals(x3d)
    scaler = StandardScaler()
    xScaled = scaler.fit_transform(xFlat)
    logger.info(f"MLP feature shape {xScaled.shape}")
    return xScaled, scaler

def encodeLabels(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y), encoder
