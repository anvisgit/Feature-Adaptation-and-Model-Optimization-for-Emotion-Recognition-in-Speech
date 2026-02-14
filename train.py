import os
import sys
import numpy as np
import json
import pickle
import logging
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import *
from data_preprocessing import loadDataMulti, augmentAudioBatchMulti
from model import createMultiBranchModel

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f'{LOGS_DIR}/train_{timestamp}.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def clearMemory():
    logger.debug("Clearing memory and resetting Keras session")
    tf.keras.backend.clear_session()
    gc.collect()

def saveResults(results, filename):
    filepath = os.path.join(MODEL_DIR, filename)
    logger.info(f"Saving results to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def saveNumpy(data, filename):
    filepath = os.path.join(MODEL_DIR, filename)
    logger.debug(f"Saving numpy array to {filepath}")
    np.save(filepath, data)

def toOneHot(y, numClasses):
    logger.debug(f"Converting {len(y)} labels to one-hot with {numClasses} classes")
    oh = np.zeros((len(y), numClasses))
    for i, label in enumerate(y):
        oh[i, label] = 1.0
    return oh

def mixupMulti(xList, y, alpha=0.2):
    logger.debug(f"Applying Mixup with alpha={alpha} to {len(y)} multi-input samples")
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

def trainMultiBranchCv(xMulti, y, audios, featParams):
    logger.info(f"Starting 5-fold StratifiedKFold cross-validation for Multi-Branch 1D-CNN")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    numClasses = len(np.unique(y))
    logger.info(f"Number of classes: {numClasses}, total samples: {len(y)}")

    classWeights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    classWeightDict = {i: w for i, w in enumerate(classWeights)}
    logger.info(f"Computed class weights: {classWeightDict}")

    foldAccs, foldF1s = [], []
    yTrueAll, yPredAll, yProbAll = [], [], []
    bestHistory = None
    bestAcc = 0

    for foldIdx, (trainIdx, testIdx) in enumerate(skf.split(xMulti[0], y)):
        foldStartTime = datetime.now()
        logger.info(f"Fold {foldIdx+1}/{N_FOLDS} started at {foldStartTime.strftime('%H:%M:%S')}")
        print(f"\n--- Fold {foldIdx+1}/{N_FOLDS} (Started: {foldStartTime.strftime('%H:%M:%S')}) ---")

        logger.info(f"Splitting data: {len(trainIdx)} train, {len(testIdx)} test")
        xTrainList = [xMulti[i][trainIdx] for i in range(3)]
        xTestList = [xMulti[i][testIdx] for i in range(3)]
        yFoldTrain = y[trainIdx]
        yFoldTest = y[testIdx]
        audiosTrainFold = [audios[i] for i in trainIdx]

        logger.info(f"Augmenting audio for fold {foldIdx+1}")
        augStartTime = datetime.now()
        xAugList, yAug = augmentAudioBatchMulti(
            audiosTrainFold, yFoldTrain,
            nmfcc=featParams.get('nmfcc', 40),
            nmels=featParams.get('nmels', 128),
            nfft=featParams.get('nfft', 2048),
            hop=featParams.get('hop', 512),
            numAug=AUGMENTATION['numAudioAugments']
        )
        augDuration = (datetime.now() - augStartTime).total_seconds()
        logger.info(f"Augmentation completed in {augDuration:.1f}s, generated {len(yAug)} samples")

        if len(yAug) > 0:
            xTrainList = [np.concatenate([xTrainList[i], xAugList[i]], axis=0) for i in range(3)]
            yTrainRaw = np.concatenate([yFoldTrain, yAug], axis=0)
        else:
            yTrainRaw = yFoldTrain

        logger.info(f"Training samples after augmentation: {len(yTrainRaw)}, Test samples: {len(yFoldTest)}")
        print(f"  Training: {len(yTrainRaw)}, Test: {len(yFoldTest)}")

        yTrainOH = toOneHot(yTrainRaw, numClasses)
        yTestOH = toOneHot(yFoldTest, numClasses)

        logger.info("Applying Mixup to training data")
        xTrainMixed, yTrainMixed = mixupMulti(xTrainList, yTrainOH, MIXUP_ALPHA)

        clearMemory()

        logger.info("Creating Multi-Branch 1D-CNN model")
        model = createMultiBranchModel(
            mfccShape=xTrainMixed[0].shape[1:],
            melShape=xTrainMixed[1].shape[1:],
            chromaShape=xTrainMixed[2].shape[1:],
            numClasses=numClasses
        )

        logger.info("Setting up cosine decay learning rate schedule")
        stepsPerEpoch = max(1, len(yTrainMixed) // BATCH_SIZE)
        totalSteps = stepsPerEpoch * N_EPOCHS
        lrSchedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=totalSteps,
            alpha=1e-7
        )
        model.compile(optimizer=Adam(learning_rate=lrSchedule),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=PATIENCE,
                restore_best_weights=True, mode='max'
            ),
        ]

        logger.info(f"Starting training for max {N_EPOCHS} epochs with batch size {BATCH_SIZE}")
        print(f"  Starting training (max {N_EPOCHS} epochs)...")
        trainStartTime = datetime.now()

        history = model.fit(
            xTrainMixed, yTrainMixed,
            epochs=N_EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(xTestList, yTestOH),
            callbacks=callbacks, verbose=1,
            class_weight=classWeightDict
        )

        trainDuration = (datetime.now() - trainStartTime).total_seconds()
        logger.info(f"Training completed in {trainDuration/60:.1f} minutes")

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

        foldDuration = (datetime.now() - foldStartTime).total_seconds()
        logger.info(f"Fold {foldIdx+1} results: accuracy={acc:.4f} f1={f1:.4f} time={foldDuration/60:.1f}min")
        print(f"  Fold {foldIdx+1} Results: Acc={acc:.4f}, F1={f1:.4f}")
        print(f"  Fold time: {foldDuration/60:.1f} minutes\n")

        del model, xTrainList, xTestList, xTrainMixed, yTrainMixed
        del xAugList, yAug
        clearMemory()

    meanAcc = np.mean(foldAccs)
    meanF1 = np.mean(foldF1s)
    stdAcc = np.std(foldAccs)
    logger.info(f"MultiBranch final: mean accuracy={meanAcc:.4f} std={stdAcc:.4f} mean f1={meanF1:.4f}")
    print(f"\n{'='*60}")
    print(f"Multi-Branch 1D-CNN Fusion Results:")
    print(f"  Mean Accuracy: {meanAcc:.4f} (+/- {stdAcc:.4f})")
    print(f"  Mean F1: {meanF1:.4f}")
    print(f"  Fold Accuracies: {[f'{a:.4f}' for a in foldAccs]}")
    print(f"{'='*60}\n")

    resKey = "multi_multibranch"
    allResults = {
        resKey: {
            'test_accuracy': float(meanAcc),
            'test_f1': float(meanF1),
            'std_accuracy': float(stdAcc),
            'fold_accuracies': [float(a) for a in foldAccs],
            'confusion_matrix': confusion_matrix(yTrueAll, yPredAll).tolist()
        }
    }

    try:
        saveNumpy(np.array(yProbAll), f'{resKey}_probs.npy')
        saveNumpy(np.array(yTrueAll), f'{resKey}_ytrue.npy')
        saveNumpy(np.array(yPredAll), f'{resKey}_ypred.npy')
        with open(os.path.join(MODEL_DIR, f'{resKey}_history.pkl'), 'wb') as f:
            pickle.dump(bestHistory, f)
        logger.info(f"Saved all results for {resKey}")
    except Exception as e:
        logger.error(f"Error saving results for {resKey}: {e}")

    return allResults

def main():
    logger.info("SER Pipeline starting - Multi-Branch 1D-CNN Feature Fusion")
    print("="*70)
    print("SER Pipeline - Multi-Branch 1D-CNN (MFCC + Mel + Chroma) Fusion")
    print("="*70)

    try:
        featureStartTime = datetime.now()
        featParams = FEATURE_PARAMS['multi']
        logger.info(f"Loading multi-feature data with params: {featParams}")

        xMulti, y, audios = loadDataMulti(
            DATASET_PATH, EMOTION_TO_ID,
            nmfcc=featParams['nmfcc'],
            nmels=featParams['nmels'],
            nfft=featParams['nfft'],
            hop=featParams['hop']
        )

        logger.info(f"Data loaded: MFCC={xMulti[0].shape} Mel={xMulti[1].shape} Chroma={xMulti[2].shape}")
        print(f"\nData loaded:")
        print(f"  MFCC shape:   {xMulti[0].shape}")
        print(f"  Mel shape:    {xMulti[1].shape}")
        print(f"  Chroma shape: {xMulti[2].shape}")
        print(f"  Labels:       {y.shape} ({len(np.unique(y))} classes)\n")

        results = trainMultiBranchCv(xMulti, y, audios, featParams)

        featureDuration = (datetime.now() - featureStartTime).total_seconds()
        logger.info(f"Pipeline completed in {featureDuration/3600:.2f} hours")
        print(f"\nPipeline completed in {featureDuration/3600:.2f} hours\n")

        saveResults(results, 'all_model_results.json')
        logger.info("Results saved to all_model_results.json")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nPipeline failed: {e}\n")
        return

    logger.info("Pipeline finished successfully")
    print("\n" + "="*70)
    print("Pipeline Finished Successfully!")
    print("="*70)

    logger.info("Generating visualizations")
    print("\nGenerating visualizations")
    try:
        import viz
        viz.generateAllVisualizations()
        logger.info("Visualizations complete")
        print("Visualizations complete")
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
