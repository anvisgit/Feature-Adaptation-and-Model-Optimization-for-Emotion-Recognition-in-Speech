import os
import sys
import numpy as np
import json
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from config import *
from data_preprocessing import loadData, augmentAudioBatch
from features import extractStatisticalFunctionals
from model import createMlpModel, createCnn1dModel, createLstmModel, createHybridModel

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f'{LOGS_DIR}/train_{timestamp}.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Add print to confirm version
print("Starting SER Pipeline Execution (Version: Fixed Imports & 4x Augmentation)...")

def saveResults(results, filename):
    with open(os.path.join(MODEL_DIR, filename), 'w') as f:
        json.dump(results, f, indent=4)

def saveNumpy(data, filename):
    np.save(os.path.join(MODEL_DIR, filename), data)

def toOneHot(y, numClasses):
    oh = np.zeros((len(y), numClasses))
    for i, label in enumerate(y):
        oh[i, label] = 1.0
    return oh

def labelSmooth(yOneHot, smoothing=0.1):
    numClasses = yOneHot.shape[1]
    return yOneHot * (1.0 - smoothing) + (smoothing / numClasses)

def mixupData(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=len(x))
    else:
        lam = np.ones(len(x))
    idx = np.random.permutation(len(x))
    if x.ndim == 3:
        lamX = lam[:, np.newaxis, np.newaxis]
    else:
        lamX = lam[:, np.newaxis]
    lamY = lam[:, np.newaxis]
    xMix = lamX * x + (1 - lamX) * x[idx]
    yMix = lamY * y + (1 - lamY) * y[idx]
    return xMix, yMix

def trainModelsCv(xSeq, y, audios, featureType, featParams):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    numClasses = len(np.unique(y))
    modelsConfig = {
        'mlp': {'func': createMlpModel, 'data': 'flat'},
        'cnn1d': {'func': createCnn1dModel, 'data': 'seq'},
        'lstm': {'func': createLstmModel, 'data': 'seq'},
        'hybrid': {'func': createHybridModel, 'data': 'seq'}
    }

    allResults = {}

    classWeights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    classWeightDict = {i: w for i, w in enumerate(classWeights)}
    logger.info(f"Class weights: {classWeightDict}")

    for modelName, cfg in modelsConfig.items():
        tf.keras.backend.clear_session()
        logger.info(f"===== Training {modelName} on {featureType} =====")
        print(f"Training {modelName} on {featureType}...")
        foldAccs, foldF1s = [], []
        yTrueAll, yPredAll, yProbAll = [], [], []
        bestHistory = None
        bestAcc = 0

        for foldIdx, (trainIdx, testIdx) in enumerate(skf.split(xSeq, y)):
            logger.info(f"Fold {foldIdx+1}/{N_FOLDS}")
            print(f"Fold {foldIdx+1}/{N_FOLDS}...")

            xFoldTrainSeq = xSeq[trainIdx]
            xFoldTestSeq = xSeq[testIdx]
            yFoldTrain = y[trainIdx]
            yFoldTest = y[testIdx]
            audiosTrainFold = [audios[i] for i in trainIdx]

            logger.info(f"Augmenting audio for fold {foldIdx+1}...")
            xAug, yAug = augmentAudioBatch(audiosTrainFold, yFoldTrain, featureType, featParams,
                                            numAug=AUGMENTATION['num_audio_augments'])
            if len(xAug) > 0:
                xTrainSeq = np.concatenate([xFoldTrainSeq, xAug], axis=0)
                yTrainRaw = np.concatenate([yFoldTrain, yAug], axis=0)
            else:
                xTrainSeq = xFoldTrainSeq
                yTrainRaw = yFoldTrain

            logger.info(f"Training samples after aug: {len(xTrainSeq)}")

            if cfg['data'] == 'flat':
                xTrain = extractStatisticalFunctionals(xTrainSeq)
                xTest = extractStatisticalFunctionals(xFoldTestSeq)
                scaler = StandardScaler()
                xTrain = scaler.fit_transform(xTrain)
                xTest = scaler.transform(xTest)
                inputShape = (xTrain.shape[1],)
            else:
                xTrain = xTrainSeq
                xTest = xFoldTestSeq
                inputShape = (xTrain.shape[1], xTrain.shape[2])

            yTrainOH = toOneHot(yTrainRaw, numClasses)
            yTrainSmoothed = labelSmooth(yTrainOH, LABEL_SMOOTHING)
            yTestOH = toOneHot(yFoldTest, numClasses)

            xTrainMix, yTrainMix = mixupData(xTrain, yTrainSmoothed, MIXUP_ALPHA)

            model = cfg['func'](inputShape=inputShape, numClasses=numClasses)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True, mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
            ]

            history = model.fit(xTrainMix, yTrainMix,
                                epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(xTest, yTestOH),
                                callbacks=callbacks, verbose=0,
                                class_weight=classWeightDict)

            if bestHistory is None or max(history.history.get('val_accuracy', [0])) > bestAcc:
                bestHistory = history.history
                bestAcc = max(history.history.get('val_accuracy', [0]))

            probs = model.predict(xTest, verbose=0)
            preds = np.argmax(probs, axis=1)
            acc = accuracy_score(yFoldTest, preds)
            f1 = f1_score(yFoldTest, preds, average='weighted')
            foldAccs.append(acc)
            foldF1s.append(f1)
            yTrueAll.extend(yFoldTest)
            yPredAll.extend(preds)
            yProbAll.extend(probs)
            logger.info(f"Fold {foldIdx+1} Acc: {acc:.4f}, F1: {f1:.4f}")
            print(f"Fold {foldIdx+1} Acc: {acc:.4f}")
            del model
            tf.keras.backend.clear_session()

        meanAcc = np.mean(foldAccs)
        meanF1 = np.mean(foldF1s)
        stdAcc = np.std(foldAccs)
        logger.info(f"{modelName} Mean Accuracy: {meanAcc:.4f} (+/- {stdAcc:.4f}), Mean F1: {meanF1:.4f}")

        resKey = f"{featureType}_{modelName}"
        allResults[resKey] = {
            'test_accuracy': float(meanAcc),
            'test_f1': float(meanF1),
            'std_accuracy': float(stdAcc),
            'fold_accuracies': [float(a) for a in foldAccs],
            'confusion_matrix': confusion_matrix(yTrueAll, yPredAll).tolist()
        }
        saveNumpy(np.array(yProbAll), f'{resKey}_probs.npy')
        saveNumpy(np.array(yTrueAll), f'{resKey}_ytrue.npy')
        saveNumpy(np.array(yPredAll), f'{resKey}_ypred.npy')
        with open(os.path.join(MODEL_DIR, f'{resKey}_history.pkl'), 'wb') as f:
            pickle.dump(bestHistory, f)

    return allResults

def main():
    logger.info("Starting High-Performance SER Pipeline")
    print("Pipeline initialized. This may take hours due to 4x Audio Augmentation per Fold.")
    featuresToRun = ['mfcc', 'mel', 'gauss']
    grandResults = {}

    for feat in featuresToRun:
        logger.info(f"========== Processing Feature: {feat} ==========")
        try:
            x, y, audios = loadData(DATASET_PATH, EMOTION_TO_ID, featType=feat, **FEATURE_PARAMS[feat])
            results = trainModelsCv(x, y, audios, feat, FEATURE_PARAMS[feat])
            grandResults.update(results)
        except Exception as e:
            logger.error(f"Failed {feat}: {e}", exc_info=True)

    saveResults(grandResults, 'all_model_results.json')
    logger.info("Pipeline Finished.")

    logger.info("Generating visualizations...")
    try:
        import viz
        viz.generateAllVisualizations()
        logger.info("Visualizations complete.")
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
