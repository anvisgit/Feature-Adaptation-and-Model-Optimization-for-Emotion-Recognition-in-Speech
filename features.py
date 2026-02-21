import numpy as np
import librosa
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis
import cv2
import logging

logger = logging.getLogger(__name__)

def teagerEnergyOperator(signal):
    logger.debug(f"Computing Teager Energy Operator on signal length {len(signal)}")
    if len(signal) < 3:
        return np.zeros_like(signal)
    te = np.zeros_like(signal)
    te[1:-1] = signal[1:-1]**2 - signal[2:] * signal[:-2]
    return te

def computeCepstralFeatures(signal, sr, nCepstral, nfft=2048, hop=512):
    logger.debug(f"Computing cepstral features with {nCepstral} coefficients")
    melSpec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40, n_fft=nfft, hop_length=hop)
    logPowerSpectrum = np.log(np.maximum(melSpec, 1e-10))
    cepstralFeatures = dct(logPowerSpectrum, type=2, axis=0, norm='ortho')[:nCepstral]
    return cepstralFeatures

def extractTeagerEnergyCepstralFeatures(y, sr, nCepstral=40, nfft=2048, hop=512):
    logger.debug("Extracting TECC features")
    teagerEnergy = teagerEnergyOperator(y)
    teccStatic = computeCepstralFeatures(teagerEnergy, sr, nCepstral, nfft, hop)
    teccDelta = librosa.feature.delta(teccStatic, order=1)
    teccDeltaDelta = librosa.feature.delta(teccStatic, order=2)
    return teccStatic.T, teccDelta.T, teccDeltaDelta.T

def extractGauss(speech, fs, windowLength=25, nfft=2048, noFilter=64):
    logger.debug(f"Extracting Gaussian filterbank features window={windowLength}ms nfft={nfft} filters={noFilter}")
    frameLengthInSample = int((fs / 1000) * windowLength)
    hopLen = frameLengthInSample // 2
    if len(speech) < frameLengthInSample:
        speech = np.pad(speech, (0, frameLengthInSample - len(speech)))
    framedSpeech = librosa.util.frame(speech, frame_length=frameLengthInSample, hop_length=hopLen)
    w = np.hanning(frameLengthInSample).reshape(-1, 1)
    yFramed = framedSpeech * w
    dftResult = np.abs(np.fft.fft(yFramed, n=nfft, axis=0))
    frAll = dftResult ** 2
    faAll = frAll[:(nfft // 2) + 1, :]
    filbanksum = gaussian_filter(faAll, sigma=2, order=0, mode='nearest', truncate=3)
    epsilon = np.finfo(float).eps
    t = dct(np.log10(filbanksum + epsilon), type=2, norm='ortho')
    t = t[1:noFilter, :]
    stat = t.T
    delta = librosa.feature.delta(stat, order=1, axis=0)
    doubleDelta = librosa.feature.delta(delta, order=1, axis=0)
    return stat, delta, doubleDelta

def extractMfcc(audio, sr, nMfcc=40, nfft=2048, hopLength=512):
    logger.debug(f"Extracting MFCCs nMfcc={nMfcc} nfft={nfft} hop={hopLength}")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=nMfcc, hop_length=hopLength, n_fft=nfft)
    delta = librosa.feature.delta(mfccs, order=1)
    deltaDelta = librosa.feature.delta(mfccs, order=2)
    return mfccs.T, delta.T, deltaDelta.T

def extractMel(audio, sr, nmels=128, nfft=2048, hop=512):
    logger.debug(f"Extracting 1D Mel spectrogram nmels={nmels}")
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=nmels, n_fft=nfft, hop_length=hop)
    melLog = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(melLog, order=1)
    deltaDelta = librosa.feature.delta(melLog, order=2)
    return melLog.T, delta.T, deltaDelta.T

def extractMelSpec2D(audio, sr, nmels=128, nfft=2048, hop=512, imgSize=224):
    logger.debug(f"Extracting 2D Mel spectrogram image nmels={nmels} imgSize={imgSize}")
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=nmels, n_fft=nfft, hop_length=hop)
    melDb = librosa.power_to_db(mel, ref=np.max)
    melNorm = (melDb - melDb.min()) / (melDb.max() - melDb.min() + 1e-8) * 255.0
    melResized = cv2.resize(melNorm, (imgSize, imgSize), interpolation=cv2.INTER_LINEAR)
    delta = librosa.feature.delta(melDb, order=1)
    deltaNorm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8) * 255.0
    deltaResized = cv2.resize(deltaNorm, (imgSize, imgSize), interpolation=cv2.INTER_LINEAR)
    delta2 = librosa.feature.delta(melDb, order=2)
    delta2Norm = (delta2 - delta2.min()) / (delta2.max() - delta2.min() + 1e-8) * 255.0
    delta2Resized = cv2.resize(delta2Norm, (imgSize, imgSize), interpolation=cv2.INTER_LINEAR)
    img3ch = np.stack([melResized, deltaResized, delta2Resized], axis=-1)
    logger.debug(f"Generated 3 channel image shape {img3ch.shape}")
    return img3ch

def extractChroma(audio, sr, nfft=2048, hop=512):
    logger.debug(f"Extracting Chroma features nfft={nfft} hop={hop}")
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=nfft, hop_length=hop)
    return chroma.T

def extractMultiFeature(audio, sr, nmfcc=40, nmels=128, nfft=2048, hop=512, ntecc=40):
    logger.debug(f"Extracting multi-feature set: MFCC({nmfcc}+deltas) + Mel({nmels}) + Chroma(12+deltas) + TECC({ntecc}+deltas)")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=nmfcc, n_fft=nfft, hop_length=hop)
    mfccDelta = librosa.feature.delta(mfcc, order=1)
    mfccDelta2 = librosa.feature.delta(mfcc, order=2)
    mfccFull = np.concatenate([mfcc, mfccDelta, mfccDelta2], axis=0).T

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=nmels, n_fft=nfft, hop_length=hop)
    melDb = librosa.power_to_db(mel, ref=np.max)
    melFull = melDb.T

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=nfft, hop_length=hop)
    chromaDelta = librosa.feature.delta(chroma, order=1)
    chromaDelta2 = librosa.feature.delta(chroma, order=2)
    chromaFull = np.concatenate([chroma, chromaDelta, chromaDelta2], axis=0).T

    teagerEnergy = teagerEnergyOperator(audio)
    teccStatic = computeCepstralFeatures(teagerEnergy, sr, ntecc, nfft, hop)
    teccDelta = librosa.feature.delta(teccStatic, order=1)
    teccDelta2 = librosa.feature.delta(teccStatic, order=2)
    teccFull = np.concatenate([teccStatic, teccDelta, teccDelta2], axis=0).T

    logger.debug(f"Multi-feature shapes: MFCC={mfccFull.shape} Mel={melFull.shape} Chroma={chromaFull.shape} TECC={teccFull.shape}")
    return mfccFull, melFull, chromaFull, teccFull

def extractFeatures(audio, sr, featType='mfcc', **kwargs):
    logger.debug(f"Extracting features type={featType}")
    if featType == 'gauss':
        return extractGauss(audio, sr, kwargs.get('window', 25), kwargs.get('nfft', 2048), kwargs.get('filters', 64))
    elif featType == 'mfcc':
        return extractMfcc(audio, sr, kwargs.get('nmfcc', 40), kwargs.get('nfft', 2048), kwargs.get('hop', 512))
    elif featType == 'teager':
        return extractTeagerEnergyCepstralFeatures(audio, sr, kwargs.get('ncep', 40))
    elif featType == 'mel':
        return extractMel(audio, sr, kwargs.get('nmels', 128), kwargs.get('nfft', 2048), kwargs.get('hop', 512))
    elif featType == 'mel2d':
        return extractMelSpec2D(audio, sr, kwargs.get('nmels', 128), kwargs.get('nfft', 2048), kwargs.get('hop', 512), kwargs.get('imgSize', 224))
    elif featType == 'multi':
        return extractMultiFeature(audio, sr, kwargs.get('nmfcc', 40), kwargs.get('nmels', 128), kwargs.get('nfft', 2048), kwargs.get('hop', 512), kwargs.get('ntecc', 40))
    else:
        raise ValueError(f"Unknown feature type: {featType}")

def extractStatisticalFunctionals(x3d):
    logger.info(f"Computing statistical functionals for {len(x3d)} samples")
    funcList = []
    for sample in x3d:
        fMean = np.mean(sample, axis=0)
        fStd = np.std(sample, axis=0)
        fMax = np.max(sample, axis=0)
        fMin = np.min(sample, axis=0)
        fMedian = np.median(sample, axis=0)
        fSkew = skew(sample, axis=0)
        fKurt = kurtosis(sample, axis=0)
        fP10 = np.percentile(sample, 10, axis=0)
        fP25 = np.percentile(sample, 25, axis=0)
        fP75 = np.percentile(sample, 75, axis=0)
        fP90 = np.percentile(sample, 90, axis=0)
        fRange = fMax - fMin
        fIqr = fP75 - fP25
        fSlope = np.polyfit(np.arange(sample.shape[0]), sample.mean(axis=1), 1)[0:1]
        fSlope = np.repeat(fSlope, sample.shape[1])[:sample.shape[1]]
        vec = np.concatenate([fMean, fStd, fMax, fMin, fMedian, fSkew, fKurt, fP10, fP25, fP75, fP90, fRange, fIqr])
        funcList.append(vec)
    logger.info(f"Statistical functionals output shape {np.array(funcList).shape}")
    return np.array(funcList)
