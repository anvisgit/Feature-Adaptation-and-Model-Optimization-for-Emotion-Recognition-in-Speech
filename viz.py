import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import librosa
import librosa.display
import os
import json
import pickle
import logging
import config
from scipy import stats
from itertools import cycle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class ResearchVisualizer:
    def __init__(self):
        self.colors = sns.color_palette('husl', len(config.EMOTION_TO_ID))
        self.emotionNames = config.EMOTION_NAMES
        logger.info("ResearchVisualizer initialized")

    def plotDatasetDistribution(self, df):
        logger.info("Creating dataset distribution visualization")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        emotionCounts = df['emotion'].value_counts()
        axes[0, 0].bar(emotionCounts.index, emotionCounts.values, color=self.colors)
        axes[0, 0].set_title('Emotion Distribution in Dataset', fontsize=16, fontweight='bold')
        axes[0, 0].set_xlabel('Emotion', fontsize=13)
        axes[0, 0].set_ylabel('Number of Samples', fontsize=13)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(emotionCounts.values):
            axes[0, 0].text(i, v + 5, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
        actorCounts = df['actor'].value_counts().sort_index()
        axes[0, 1].bar(range(len(actorCounts)), actorCounts.values, color='skyblue', alpha=0.8)
        axes[0, 1].set_title('Samples per Actor', fontsize=16, fontweight='bold')
        axes[0, 1].set_xlabel('Actor ID', fontsize=13)
        axes[0, 1].set_ylabel('Number of Samples', fontsize=13)
        axes[0, 1].axhline(actorCounts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {actorCounts.mean():.1f}')
        axes[0, 1].legend()
        emotionActor = pd.crosstab(df['emotion'], df['actor'])
        sns.heatmap(emotionActor, annot=False, cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Count'})
        axes[1, 0].set_title('Emotion Distribution Across Actors', fontsize=16, fontweight='bold')
        axes[1, 0].set_xlabel('Actor ID', fontsize=13)
        axes[1, 0].set_ylabel('Emotion', fontsize=13)
        emotionPercentages = (emotionCounts / emotionCounts.sum()) * 100
        axes[1, 1].pie(emotionPercentages, labels=emotionPercentages.index, autopct='%1.1f%%',
                      colors=self.colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[1, 1].set_title('Emotion Distribution (Percentage)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        savePath = os.path.join(config.VIZ_DIR, 'dataset_distribution.png')
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Dataset distribution saved to {savePath}")

    def plotAudioFeaturesAnalysis(self, df, numSamples=8):
        logger.info("Creating audio features analysis visualization")
        fig, axes = plt.subplots(len(config.EMOTION_TO_ID), numSamples, figsize=(24, 20))
        for emotionIdx, emotion in enumerate(list(config.EMOTION_TO_ID.keys())):
            emotionSamples = df[df['emotion'] == emotion].sample(min(numSamples, len(df[df['emotion'] == emotion])))
            for sampleIdx, (_, row) in enumerate(emotionSamples.iterrows()):
                if sampleIdx >= numSamples: break
                try:
                    audio, sr = librosa.load(row['path'], sr=config.SAMPLE_RATE, duration=3)
                    melSpec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=config.N_MELS,
                                                              n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
                    melSpecDb = librosa.power_to_db(melSpec, ref=np.max)
                    librosa.display.specshow(melSpecDb, sr=sr, hop_length=config.HOP_LENGTH,
                                                   x_axis='time', y_axis='mel', ax=axes[emotionIdx, sampleIdx],
                                                   cmap='viridis')
                    if sampleIdx == 0:
                        axes[emotionIdx, sampleIdx].set_ylabel(f'{emotion.capitalize()}',
                                                                  fontsize=12, fontweight='bold')
                    else:
                        axes[emotionIdx, sampleIdx].set_ylabel('')
                    if emotionIdx == 0:
                        axes[emotionIdx, sampleIdx].set_title(f'Sample {sampleIdx+1}', fontsize=11)
                    axes[emotionIdx, sampleIdx].set_xlabel('')
                except Exception as e:
                    logger.error(f"Error processing {row['path']}: {str(e)}")
                    axes[emotionIdx, sampleIdx].axis('off')
        plt.suptitle('Mel-Spectrogram Analysis Across Emotions', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        savePath = os.path.join(config.VIZ_DIR, 'audio_features_mel_spectrograms.png')
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Audio features analysis saved to {savePath}")

    def plotTrainingHistory(self, modelName):
        logger.info(f"Creating training history for {modelName}")
        historyPath = os.path.join(config.MODEL_DIR, f'{modelName}_history.pkl')
        if not os.path.exists(historyPath): return
        with open(historyPath, 'rb') as f: history = pickle.load(f)
        if not isinstance(history, dict): return
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(history.get('loss', [])) + 1)
        if not epochs: return
        axes[0].plot(epochs, history.get('loss'), label='Train', linewidth=2)
        axes[0].plot(epochs, history.get('val_loss'), label='Val', linewidth=2)
        axes[0].set_title(f'Loss - {modelName}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, history.get('accuracy'), label='Train', linewidth=2)
        axes[1].plot(epochs, history.get('val_accuracy'), label='Val', linewidth=2)
        axes[1].set_title(f'Accuracy - {modelName}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.VIZ_DIR, f'training_history_{modelName}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plotConfusionMatrix(self, modelName, yTrue, yPred, normalize=False):
        logger.info(f"Creating CM for {modelName}")
        cm = confusion_matrix(yTrue, yPred, labels=range(len(self.emotionNames)))
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=self.emotionNames, yticklabels=self.emotionNames)
        plt.title(f"{'Normalized ' if normalize else ''}Confusion Matrix: {modelName}", fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(config.VIZ_DIR, f'confusion_matrix_{modelName}{"_norm" if normalize else ""}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plotClassificationReportHeatmap(self, modelName, yTrue, yPred):
        logger.info(f"Creating classification report heatmap for {modelName}")
        clfReport = classification_report(yTrue, yPred, labels=range(len(self.emotionNames)), target_names=self.emotionNames, output_dict=True, zero_division=0)
        reportDf = pd.DataFrame(clfReport).iloc[:-1, :].T
        plt.figure(figsize=(12, 8))
        sns.heatmap(reportDf, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title(f'Classification Report: {modelName}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.VIZ_DIR, f'classification_metrics_{modelName}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plotRocCurves(self, modelName, yTrue, yPredProbs):
        logger.info(f"Creating ROC for {modelName}")
        yBin = label_binarize(yTrue, classes=range(len(self.emotionNames)))
        nClasses = yBin.shape[1]
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(yBin.ravel(), yPredProbs.ravel())
        aucScore = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Micro-average (AUC={aucScore:.3f})', linewidth=2)
        for i in range(nClasses):
            f, t, _ = roc_curve(yBin[:, i], yPredProbs[:, i])
            a = auc(f, t)
            plt.plot(f, t, label=f'{self.emotionNames[i]} (AUC={a:.3f})', alpha=0.7)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.legend(loc='lower right', fontsize=9)
        plt.title(f'ROC Curves: {modelName}', fontsize=14, fontweight='bold')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(config.VIZ_DIR, f'roc_{modelName}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plotModelComparison(self, allResults):
        logger.info("Creating model comparison")
        names, accs, f1s = [], [], []
        for k, v in allResults.items():
            names.append(k)
            accs.append(v.get('test_accuracy', 0))
            f1s.append(v.get('test_f1', 0))
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        xPos = np.arange(len(names))
        axes[0].barh(xPos, accs, color=sns.color_palette('husl', len(names)))
        axes[0].set_yticks(xPos)
        axes[0].set_yticklabels(names, fontsize=10)
        axes[0].set_xlabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlim(0, 1)
        for i, v in enumerate(accs):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
        axes[1].barh(xPos, f1s, color=sns.color_palette('husl', len(names)))
        axes[1].set_yticks(xPos)
        axes[1].set_yticklabels(names, fontsize=10)
        axes[1].set_xlabel('F1 Score', fontsize=12)
        axes[1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)
        for i, v in enumerate(f1s):
            axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.VIZ_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generateAllVisualizations():
    viz = ResearchVisualizer()
    import data_preprocessing
    df = data_preprocessing.loadDataset(config.DATASET_PATH)

    if not df.empty:
        viz.plotDatasetDistribution(df)
        viz.plotAudioFeaturesAnalysis(df)

    resPath = os.path.join(config.MODEL_DIR, 'all_model_results.json')
    if os.path.exists(resPath):
        with open(resPath, 'r') as f: results = json.load(f)
        viz.plotModelComparison(results)

        for modelKey in results.keys():
            viz.plotTrainingHistory(modelKey)
            try:
                yTrue = np.load(os.path.join(config.MODEL_DIR, f'{modelKey}_ytrue.npy'))
                yPred = np.load(os.path.join(config.MODEL_DIR, f'{modelKey}_ypred.npy'))
                yProbs = np.load(os.path.join(config.MODEL_DIR, f'{modelKey}_probs.npy'))
                viz.plotConfusionMatrix(modelKey, yTrue, yPred)
                viz.plotConfusionMatrix(modelKey, yTrue, yPred, normalize=True)
                viz.plotRocCurves(modelKey, yTrue, yProbs)
                viz.plotClassificationReportHeatmap(modelKey, yTrue, yPred)
            except Exception as e:
                logger.error(f"Visualization skip {modelKey}: {e}")

if __name__ == "__main__":
    generateAllVisualizations()
