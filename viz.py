import numpy as np
import os
import pickle
import json
import logging
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import config

logger = logging.getLogger(__name__)

sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams.update({
    'figure.figsize': (14, 10), 'font.size': 11, 'axes.labelsize': 13,
    'axes.titlesize': 15, 'axes.titleweight': 'bold', 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2,
})

emotionNames = config.EMOTION_NAMES
emotionColors = sns.color_palette('husl', len(emotionNames))
vizDir = config.VIZ_DIR
modelDir = config.MODEL_DIR


def getSamplePerEmotion(datasetPath, emotionMap):
    logger.info("Scanning dataset for samples")
    samples = {}
    for subdir, dirs, files in os.walk(datasetPath):
        for f in files:
            if f.endswith('.wav'):
                parts = f.split('.')[0].split('-')
                if len(parts) >= 3 and parts[2] in emotionMap:
                    emId = emotionMap[parts[2]]
                    if emId not in samples:
                        samples[emId] = os.path.join(subdir, f)
                        if len(samples) == len(emotionMap):
                            return samples
    return samples


def plotAudioFeaturesAnalysis():
    logger.info("Generating audio features analysis")
    samples = getSamplePerEmotion(config.DATASET_PATH, config.EMOTION_TO_ID)
    if len(samples) < 4:
        logger.warning("Insufficient samples")
        return

    emotionsToShow = [3, 4, 5, 7]
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(len(emotionsToShow), 3, hspace=0.45, wspace=0.3)

    for row, emId in enumerate(emotionsToShow):
        if emId not in samples:
            continue
        audio, sr = librosa.load(samples[emId], sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)

        ax1 = fig.add_subplot(gs[row, 0])
        timeAxis = np.linspace(0, len(audio) / sr, len(audio))
        ax1.plot(timeAxis, audio, color=emotionColors[emId], linewidth=0.5, alpha=0.8)
        ax1.fill_between(timeAxis, audio, alpha=0.15, color=emotionColors[emId])
        ax1.set_title(f'{emotionNames[emId]} — Waveform', fontsize=11)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim(0, config.AUDIO_DURATION)

        ax2 = fig.add_subplot(gs[row, 1])
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        melDb = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(melDb, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
        ax2.set_title(f'{emotionNames[emId]} — Mel Spectrogram', fontsize=11)
        fig.colorbar(img, ax=ax2, format='%+2.0f dB', shrink=0.8)

        ax3 = fig.add_subplot(gs[row, 2])
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        img2 = librosa.display.specshow(mfccs, sr=sr, hop_length=512, x_axis='time', ax=ax3, cmap='coolwarm')
        ax3.set_title(f'{emotionNames[emId]} — MFCC (40 coefficients)', fontsize=11)
        ax3.set_ylabel('MFCC Coefficient')
        fig.colorbar(img2, ax=ax3, shrink=0.8)

    fig.suptitle('Audio Features Analysis: Waveform, Mel Spectrogram & MFCC', fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(vizDir, '01_audio_features_analysis.png'))
    plt.close()
    logger.info("Saved 01_audio_features_analysis")


def plotSpectrogramComparison():
    logger.info("Generating spectrogram comparison")
    samples = getSamplePerEmotion(config.DATASET_PATH, config.EMOTION_TO_ID)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for emId in range(8):
        ax = axes[emId]
        if emId not in samples:
            ax.set_title(f'{emotionNames[emId]} — No Sample')
            continue
        audio, sr = librosa.load(samples[emId], sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        melDb = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(melDb, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='inferno')
        ax.set_title(f'{emotionNames[emId]}', fontsize=13, fontweight='bold', color=emotionColors[emId])
        if emId % 4 != 0:
            ax.set_ylabel('')
        if emId < 4:
            ax.set_xlabel('')

    fig.suptitle('Mel Spectrogram Comparison Across All 8 Emotions', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(vizDir, '02_spectrogram_comparison.png'))
    plt.close()
    logger.info("Saved 02_spectrogram_comparison")


def plotPitchEnergyContours():
    logger.info("Generating pitch/energy contours")
    samples = getSamplePerEmotion(config.DATASET_PATH, config.EMOTION_TO_ID)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    for emId in range(len(emotionNames)):
        if emId not in samples:
            continue
        audio, sr = librosa.load(samples[emId], sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, n_fft=2048, hop_length=512)
        pitchTrack = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitchTrack.append(pitches[idx, t])
        pitchTrack = np.array(pitchTrack)
        pitchSmooth = np.convolve(pitchTrack, np.ones(5) / 5, mode='same')
        pitchSmooth[pitchSmooth < 50] = np.nan
        timeAxis = np.linspace(0, config.AUDIO_DURATION, len(pitchSmooth))
        ax1.plot(timeAxis, pitchSmooth, color=emotionColors[emId], linewidth=1.5,
                 label=emotionNames[emId], alpha=0.8)

        rms = librosa.feature.rms(y=audio, hop_length=512)[0]
        rmsDb = librosa.amplitude_to_db(rms, ref=np.max)
        timeRms = np.linspace(0, config.AUDIO_DURATION, len(rmsDb))
        ax2.plot(timeRms, rmsDb, color=emotionColors[emId], linewidth=1.5,
                 label=emotionNames[emId], alpha=0.8)

    ax1.set_title('Fundamental Frequency (F0) Contour per Emotion', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.legend(loc='upper right', ncol=4, fontsize=9)
    ax1.set_xlim(0, config.AUDIO_DURATION)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Energy (RMS) Contour per Emotion', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (dB)')
    ax2.legend(loc='upper right', ncol=4, fontsize=9)
    ax2.set_xlim(0, config.AUDIO_DURATION)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Prosodic Feature Analysis: Pitch & Energy', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(vizDir, '03_pitch_energy_contours.png'))
    plt.close()
    logger.info("Saved 03_pitch_energy_contours")


def plotFeatureCorrelation():
    logger.info("Generating feature correlation heatmap")
    samples = getSamplePerEmotion(config.DATASET_PATH, config.EMOTION_TO_ID)
    allMfccs = []
    allLabels = []

    for emId in range(len(emotionNames)):
        if emId not in samples:
            continue
        audio, sr = librosa.load(samples[emId], sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=512)
        meanMfcc = np.mean(mfccs, axis=1)
        allMfccs.append(meanMfcc)
        allLabels.append(emotionNames[emId])

    mfccMatrix = np.array(allMfccs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(mfccMatrix, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                xticklabels=[f'C{i}' for i in range(20)], yticklabels=allLabels,
                ax=ax1, linewidths=0.3)
    ax1.set_title('Mean MFCC Values per Emotion', fontweight='bold')
    ax1.set_xlabel('MFCC Coefficient')
    ax1.set_ylabel('Emotion')

    corrMatrix = np.corrcoef(mfccMatrix)
    mask = np.triu(np.ones_like(corrMatrix, dtype=bool), k=1)
    sns.heatmap(corrMatrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=allLabels, yticklabels=allLabels, ax=ax2,
                vmin=-1, vmax=1, linewidths=0.5, center=0)
    ax2.set_title('Inter-Emotion MFCC Correlation', fontweight='bold')

    fig.suptitle('MFCC Feature Analysis & Emotion Correlation', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(vizDir, '04_feature_correlation.png'))
    plt.close()
    logger.info("Saved 04_feature_correlation")


def plotChromaComparison():
    logger.info("Generating chroma comparison")
    samples = getSamplePerEmotion(config.DATASET_PATH, config.EMOTION_TO_ID)
    emotionsToShow = [0, 2, 4, 6]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, emId in enumerate(emotionsToShow):
        ax = axes[i]
        if emId not in samples:
            continue
        audio, sr = librosa.load(samples[emId], sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512)
        img = librosa.display.specshow(chroma, sr=sr, hop_length=512, x_axis='time', y_axis='chroma',
                                       ax=ax, cmap='YlOrRd')
        ax.set_title(f'{emotionNames[emId]}', fontsize=13, fontweight='bold', color=emotionColors[emId])
        fig.colorbar(img, ax=ax, shrink=0.8)

    fig.suptitle('Chromagram Comparison — Tonal Content per Emotion', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(vizDir, '05_chroma_comparison.png'))
    plt.close()
    logger.info("Saved 05_chroma_comparison")


def plotDimensionalityReduction():
    logger.info("Generating t-SNE visualization")
    from data_preprocessing import loadDataMulti

    try:
        xMulti, y, _ = loadDataMulti(
            config.DATASET_PATH, config.EMOTION_TO_ID,
            nmfcc=40, nmels=128, nfft=2048, hop=512
        )
    except Exception as e:
        logger.error(f"t-SNE data load failed: {e}")
        return

    xMfccMean = np.mean(xMulti[0], axis=1)
    xMelMean = np.mean(xMulti[1], axis=1)
    xChromaMean = np.mean(xMulti[2], axis=1)
    xCombined = np.concatenate([xMfccMean, xMelMean, xChromaMean], axis=1)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    xEmbedded = tsne.fit_transform(xCombined)

    fig, ax = plt.subplots(figsize=(14, 10))
    for emId in range(len(emotionNames)):
        mask = y == emId
        ax.scatter(xEmbedded[mask, 0], xEmbedded[mask, 1],
                   c=[emotionColors[emId]], label=emotionNames[emId],
                   alpha=0.65, s=40, edgecolors='white', linewidths=0.3)
    ax.set_title('t-SNE of Multi-Feature Space (MFCC + Mel + Chroma)', fontsize=15)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vizDir, '06_tsne_dimensionality_reduction.png'))
    plt.close()
    logger.info("Saved 06_tsne_dimensionality_reduction")


def plotTrainingHistory():
    logger.info("Generating training history")
    historyPath = os.path.join(modelDir, 'multi_multibranch_history.pkl')
    if not os.path.exists(historyPath):
        logger.warning("History file not found")
        return

    with open(historyPath, 'rb') as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.85)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.85)
    ax1.fill_between(epochs, history['loss'], history['val_loss'], alpha=0.1, color='purple')
    minValLoss = min(history['val_loss'])
    minEpoch = history['val_loss'].index(minValLoss) + 1
    ax1.axvline(x=minEpoch, color='green', linestyle='--', alpha=0.6, label=f'Best ({minEpoch})')
    ax1.scatter([minEpoch], [minValLoss], color='green', s=100, zorder=5, marker='*')
    ax1.set_title('Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Train Acc', alpha=0.85)
    ax2.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Val Acc', alpha=0.85)
    ax2.fill_between(epochs, history['accuracy'], history['val_accuracy'], alpha=0.1, color='purple')
    maxValAcc = max(history['val_accuracy'])
    maxEpoch = history['val_accuracy'].index(maxValAcc) + 1
    ax2.axvline(x=maxEpoch, color='green', linestyle='--', alpha=0.6, label=f'Best ({maxEpoch})')
    ax2.scatter([maxEpoch], [maxValAcc], color='green', s=100, zorder=5, marker='*')
    ax2.annotate(f'{maxValAcc:.4f}', (maxEpoch, maxValAcc), textcoords='offset points',
                 xytext=(10, 10), fontsize=12, fontweight='bold', color='green')
    ax2.set_title('Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Training History — Multi-Branch 1D-CNN', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(vizDir, '07_training_history.png'))
    plt.close()
    logger.info("Saved 07_training_history")


def plotConfusionMatrix():
    logger.info("Generating confusion matrix")
    yTruePath = os.path.join(modelDir, 'multi_multibranch_ytrue.npy')
    yPredPath = os.path.join(modelDir, 'multi_multibranch_ypred.npy')
    if not os.path.exists(yTruePath) or not os.path.exists(yPredPath):
        logger.warning("Prediction files not found")
        return

    yTrue = np.load(yTruePath)
    yPred = np.load(yPredPath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    cm = confusion_matrix(yTrue, yPred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotionNames,
                yticklabels=emotionNames, ax=ax1, linewidths=0.5)
    ax1.set_title('Raw Counts', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmNorm, annot=True, fmt='.2%', cmap='YlOrRd', xticklabels=emotionNames,
                yticklabels=emotionNames, ax=ax2, vmin=0, vmax=1, linewidths=0.5)
    ax2.set_title('Normalized (Recall)', fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    totalAcc = np.trace(cm) / cm.sum()
    fig.suptitle(f'Confusion Matrix — Accuracy: {totalAcc:.2%}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(vizDir, '08_confusion_matrix.png'))
    plt.close()
    logger.info("Saved 08_confusion_matrix")


def plotRocCurves():
    logger.info("Generating ROC curves")
    yTruePath = os.path.join(modelDir, 'multi_multibranch_ytrue.npy')
    yProbPath = os.path.join(modelDir, 'multi_multibranch_probs.npy')
    if not os.path.exists(yTruePath) or not os.path.exists(yProbPath):
        logger.warning("Probability files not found")
        return

    yTrue = np.load(yTruePath)
    yProb = np.load(yProbPath)
    numClasses = len(emotionNames)
    yBin = label_binarize(yTrue, classes=list(range(numClasses)))

    fig, ax = plt.subplots(figsize=(12, 10))
    aucScores = []

    for i in range(numClasses):
        fpr, tpr, _ = roc_curve(yBin[:, i], yProb[:, i])
        rocAuc = auc(fpr, tpr)
        aucScores.append(rocAuc)
        ax.plot(fpr, tpr, color=emotionColors[i], linewidth=2.2,
                label=f'{emotionNames[i]} (AUC={rocAuc:.3f})', alpha=0.85)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    meanAuc = np.mean(aucScores)
    ax.set_title(f'ROC Curves — Mean AUC={meanAuc:.3f}', fontsize=15, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vizDir, '09_roc_curves.png'))
    plt.close()
    logger.info("Saved 09_roc_curves")


def plotPerEmotionPerformance():
    logger.info("Generating per-emotion metrics")
    yTruePath = os.path.join(modelDir, 'multi_multibranch_ytrue.npy')
    yPredPath = os.path.join(modelDir, 'multi_multibranch_ypred.npy')
    if not os.path.exists(yTruePath) or not os.path.exists(yPredPath):
        logger.warning("Prediction files not found")
        return

    yTrue = np.load(yTruePath)
    yPred = np.load(yPredPath)
    report = classification_report(yTrue, yPred, target_names=emotionNames, output_dict=True)

    emotions = emotionNames
    precision = [report[e]['precision'] for e in emotions]
    recall = [report[e]['recall'] for e in emotions]
    f1 = [report[e]['f1-score'] for e in emotions]

    x = np.arange(len(emotions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c', alpha=0.85, edgecolor='white')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71', alpha=0.85, edgecolor='white')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

    wf1 = report['weighted avg']['f1-score']
    ax.axhline(y=wf1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Weighted F1={wf1:.3f}')
    ax.set_title('Per-Emotion Classification Metrics', fontsize=15, fontweight='bold')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=25, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(vizDir, '10_per_emotion_performance.png'))
    plt.close()
    logger.info("Saved 10_per_emotion_performance")


def generatePreTrainingVisualizations():
    logger.info("Generating pre-training visualizations")
    os.makedirs(vizDir, exist_ok=True)

    preTrainViz = [
        ("Audio Features", plotAudioFeaturesAnalysis),
        ("Spectrogram Comparison", plotSpectrogramComparison),
        ("Pitch & Energy", plotPitchEnergyContours),
        ("Feature Correlation", plotFeatureCorrelation),
        ("Chroma Comparison", plotChromaComparison),
        ("t-SNE", plotDimensionalityReduction),
    ]

    for i, (name, func) in enumerate(preTrainViz, 1):
        try:
            logger.info(f"[{i}/{len(preTrainViz)}] {name}")
            print(f"  [{i}/{len(preTrainViz)}] {name}...")
            func()
            print(f"         Done!")
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
            print(f"         FAILED: {e}")

    logger.info("Pre-training visualizations complete")


def generatePostTrainingVisualizations():
    logger.info("Generating post-training visualizations")
    os.makedirs(vizDir, exist_ok=True)

    postTrainViz = [
        ("Training History", plotTrainingHistory),
        ("Confusion Matrix", plotConfusionMatrix),
        ("ROC Curves", plotRocCurves),
        ("Per-Emotion Metrics", plotPerEmotionPerformance),
    ]

    for i, (name, func) in enumerate(postTrainViz, 1):
        try:
            logger.info(f"[{i}/{len(postTrainViz)}] {name}")
            print(f"  [{i}/{len(postTrainViz)}] {name}...")
            func()
            print(f"         Done!")
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
            print(f"         FAILED: {e}")

    logger.info("Post-training visualizations complete")


def generateAllVisualizations():
    logger.info("Generating all visualizations")
    generatePreTrainingVisualizations()
    generatePostTrainingVisualizations()
    logger.info("All visualizations complete")
    print(f"\nAll saved to: {vizDir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    generateAllVisualizations()
