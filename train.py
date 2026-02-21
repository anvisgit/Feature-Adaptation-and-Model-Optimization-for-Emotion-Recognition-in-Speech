import os
import sys
import numpy as np
import json
import pickle
import logging
import gc
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import *
from data_preprocessing import loadDataMulti, augmentAudioBatchMulti
from model import createMultiBranchModel

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f'{LOGS_DIR}/train_{timestamp}.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

NUM_BRANCHES = 4

def clearMemory():
    tf.keras.backend.clear_session()
    gc.collect()

def saveResults(results, filename):
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def saveNumpy(data, filename):
    np.save(os.path.join(MODEL_DIR, filename), data)

def toOneHot(y, numClasses):
    oh = np.zeros((len(y), numClasses))
    for i, label in enumerate(y):
        oh[i, label] = 1.0
    return oh

def normalizeFeatures(xTrain, xTest):
    nTrain, T, F = xTrain.shape
    nTest = xTest.shape[0]
    scaler = StandardScaler()
    scaler.fit(xTrain.reshape(-1, F))
    xTrainNorm = scaler.transform(xTrain.reshape(-1, F)).reshape(nTrain, T, F).astype(np.float32)
    xTestNorm = scaler.transform(xTest.reshape(-1, F)).reshape(nTest, T, F).astype(np.float32)
    return xTrainNorm, xTestNorm

def mixupMulti(xList, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=len(y))
    else:
        lam = np.ones(len(y))
    idx = np.random.permutation(len(y))
    lamX = lam[:, np.newaxis, np.newaxis]
    lamY = lam[:, np.newaxis]
    xMixed = []
    for x in xList:
        xMixed.append(lamX * x + (1 - lamX) * x[idx])
    yMixed = lamY * y + (1 - lamY) * y[idx]
    return xMixed, yMixed

def trainMultiBranchCv(xMulti, y, audios, actorIds, featParams):
    logger.info("Starting 5-fold GroupKFold (Speaker-Independent) CV")
    gkf = GroupKFold(n_splits=N_FOLDS)
    numClasses = len(np.unique(y))

    foldAccs, foldF1s = [], []
    yTrueAll, yPredAll, yProbAll = [], [], []
    bestHistory = None
    bestAcc = 0

    for foldIdx, (trainIdx, testIdx) in enumerate(gkf.split(xMulti[0], y, groups=actorIds)):
        foldStartTime = datetime.now()
        logger.info(f"Fold {foldIdx+1}/{N_FOLDS} started")
        print(f"\nFold {foldIdx+1}/{N_FOLDS} (Speaker-Independent)")

        xTrainList = [xMulti[i][trainIdx] for i in range(NUM_BRANCHES)]
        xTestList = [xMulti[i][testIdx] for i in range(NUM_BRANCHES)]
        yFoldTrain = y[trainIdx]
        yFoldTest = y[testIdx]
        audiosTrainFold = [audios[i] for i in trainIdx]

        logger.info("Augmenting fold (4x, 8x for Neutral)")
        xAugList, yAug = augmentAudioBatchMulti(
            audiosTrainFold, yFoldTrain,
            nmfcc=featParams.get('nmfcc', 40), nmels=featParams.get('nmels', 128),
            nfft=featParams.get('nfft', 2048), hop=featParams.get('hop', 512),
            ntecc=featParams.get('ntecc', 40),
            numAug=AUGMENTATION['numAudioAugments']
        )

        if len(yAug) > 0:
            xTrainList = [np.concatenate([xTrainList[i], xAugList[i]], axis=0) for i in range(NUM_BRANCHES)]
            yTrainRaw = np.concatenate([yFoldTrain, yAug], axis=0)
        else:
            yTrainRaw = yFoldTrain

        logger.info("Normalizing features per fold")
        for i in range(NUM_BRANCHES):
            xTrainList[i], xTestList[i] = normalizeFeatures(xTrainList[i], xTestList[i])

        yTrainOH = toOneHot(yTrainRaw, numClasses)
        yTestOH = toOneHot(yFoldTest, numClasses)

        logger.info("Applying Mixup")
        xTrainMixed, yTrainMixed = mixupMulti(xTrainList, yTrainOH, MIXUP_ALPHA)

        clearMemory()

        model = createMultiBranchModel(
            mfccShape=xTrainMixed[0].shape[1:],
            melShape=xTrainMixed[1].shape[1:],
            chromaShape=xTrainMixed[2].shape[1:],
            teccShape=xTrainMixed[3].shape[1:],
            numClasses=numClasses,
            learningRate=LEARNING_RATE,
            focalGamma=FOCAL_GAMMA,
            focalAlpha=FOCAL_ALPHA
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        logger.info(f"Training Fold {foldIdx+1}")
        history = model.fit(
            xTrainMixed, yTrainMixed,
            epochs=N_EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(xTestList, yTestOH),
            callbacks=callbacks, verbose=1
        )

        if bestHistory is None or max(history.history.get('val_accuracy', [0])) > bestAcc:
            bestHistory = history.history
            bestAcc = max(history.history.get('val_accuracy', [0]))

        probs = model.predict(xTestList, verbose=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(yFoldTest, preds)
        f1 = f1_score(yFoldTest, preds, average='weighted')
        
        foldAccs.append(acc)
        foldF1s.append(f1)
        yTrueAll.extend(yFoldTest)
        yPredAll.extend(preds)
        yProbAll.extend(probs)
        
        elapsed = (datetime.now() - foldStartTime).total_seconds() / 60
        logger.info(f"Fold {foldIdx+1} done Acc={acc:.4f} F1={f1:.4f} Time={elapsed:.1f}min")
        print(f"  Acc={acc:.4f}, F1={f1:.4f}, Time={elapsed:.1f}min")
        clearMemory()

    meanAcc = np.mean(foldAccs)
    meanF1 = np.mean(foldF1s)
    stdAcc = np.std(foldAccs)
    print(f"\nFinal Mean Accuracy: {meanAcc:.4f} (+/- {stdAcc:.4f})")
    logger.info(f"Done Acc={meanAcc:.4f} std={stdAcc:.4f} F1={meanF1:.4f}")

    resKey = "fusion_multibranch_adv"
    allResults = {
        resKey: {
            'testAccuracy': float(meanAcc),
            'testF1': float(meanF1),
            'stdAccuracy': float(stdAcc),
            'foldAccuracies': [float(a) for a in foldAccs],
            'confusionMatrix': confusion_matrix(yTrueAll, yPredAll).tolist()
        }
    }
    
    saveResults(allResults, 'allModelResults.json')
    saveNumpy(np.array(yProbAll), f'{resKey}Probs.npy')
    saveNumpy(np.array(yTrueAll), f'{resKey}Ytrue.npy')
    saveNumpy(np.array(yPredAll), f'{resKey}Ypred.npy')

    if bestHistory:
        with open(os.path.join(MODEL_DIR, 'bestHistory.pkl'), 'wb') as f:
            pickle.dump(bestHistory, f)

    return allResults

def main():
    try:
        featParams = FEATURE_PARAMS['multi']
        xMulti, y, audios, actorIds = loadDataMulti(
            DATASET_PATH, EMOTION_TO_ID,
            nmfcc=featParams['nmfcc'], nmels=featParams['nmels'],
            nfft=featParams['nfft'], hop=featParams['hop'],
            ntecc=featParams.get('ntecc', 40)
        )
        logger.info(f"Data shapes: MFCC={xMulti[0].shape} Mel={xMulti[1].shape} Chroma={xMulti[2].shape} TECC={xMulti[3].shape}")
        print(f"Data Loaded: MFCC={xMulti[0].shape} Mel={xMulti[1].shape} Chroma={xMulti[2].shape} TECC={xMulti[3].shape}")
        print(f"Unique Actors: {len(np.unique(actorIds))}, Samples: {len(y)}")
        trainMultiBranchCv(xMulti, y, audios, actorIds, featParams)
        
        import viz
        viz.generateAllVisualizations()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
