import numpy as np
import librosa
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis

def teagerEnergyOperator(signal):
    if len(signal) < 3: return np.zeros_like(signal)
    te = np.zeros_like(signal)
    te[1:-1] = signal[1:-1]**2 - signal[2:] * signal[:-2]
    return te

def computeCepstralFeatures(signal, sr, nCepstral):
    melSpec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40)
    logPowerSpectrum = np.log(np.maximum(melSpec, 1e-10))
    cepstralFeatures = dct(logPowerSpectrum, type=2, axis=-1, norm='ortho')[:nCepstral]
    return cepstralFeatures

def extractTeagerEnergyCepstralFeatures(y, sr, nCepstral=13):
    teagerEnergy = teagerEnergyOperator(y)
    teccStatic = computeCepstralFeatures(teagerEnergy, sr, nCepstral)
    teccDelta = librosa.feature.delta(teccStatic, order=1)
    teccDeltaDelta = librosa.feature.delta(teccStatic, order=2)
    return teccStatic.T, teccDelta.T, teccDeltaDelta.T

def extractGauss(speech, fs, windowLength=25, nfft=2048, noFilter=64):
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
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=nMfcc, hop_length=hopLength, n_fft=nfft)
    delta = librosa.feature.delta(mfccs, order=1)
    deltaDelta = librosa.feature.delta(mfccs, order=2)
    return mfccs.T, delta.T, deltaDelta.T

def extractMel(audio, sr, nmels=128, nfft=2048, hop=512):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=nmels, n_fft=nfft, hop_length=hop)
    melLog = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(melLog, order=1)
    deltaDelta = librosa.feature.delta(melLog, order=2)
    return melLog.T, delta.T, deltaDelta.T

def extractFeatures(audio, sr, featType='mfcc', **kwargs):
    if featType == 'gauss':
        return extractGauss(audio, sr, kwargs.get('window', 25), kwargs.get('nfft', 2048), kwargs.get('filters', 64))
    elif featType == 'mfcc':
        return extractMfcc(audio, sr, kwargs.get('nmfcc', 40), kwargs.get('nfft', 2048), kwargs.get('hop', 512))
    elif featType == 'teager':
        return extractTeagerEnergyCepstralFeatures(audio, sr, kwargs.get('ncep', 40))
    elif featType == 'mel':
        return extractMel(audio, sr, kwargs.get('nmels', 128), kwargs.get('nfft', 2048), kwargs.get('hop', 512))
    else:
        raise ValueError(f"Unknown feature type: {featType}")

def extractStatisticalFunctionals(x3d):
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
    return np.array(funcList)
