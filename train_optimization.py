import os 
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
import config as cfg
import data_processing as dp 
import models 
import visualizations as v
import logging 

logger = logging.getLogger(__name__)

np.random.seed(42)
tf.random.set_seed(42)

def prepare_data():
    logger.info("Preparing data")
    x, y = dp.load_data()
    if x.shape[0] == 0:
        raise ValueError("No data found")
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
    
    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest) 
    
    joblib.dump(sc, os.path.join(cfg.MODELS_PATH, 'scaler.pkl'))
    joblib.dump(le, os.path.join(cfg.MODELS_PATH, 'label_encoder.pkl'))
    
    logger.info(f"Data split. Train: {xtrain.shape}, Test: {xtest.shape}")
    logger.info(f"Classes encoded: {list(zip(range(len(le.classes_)), le.classes_))}")

    return xtrain, xtest, ytrain, ytest, le

def reshape1(x): # reshape for cnn lstm ;2D->3D
    return x.reshape(x.shape[0], x.shape[1], 1)

def evaluate():
    trainx2d, testx2d, trainy, testy, le = prepare_data()
    num = len(np.unique(trainy))
    input_shape_2d = (trainx2d.shape[1],)
    
    trainx3d = reshape1(trainx2d)
    testx3d = reshape1(testx2d)
    input_shape_3d = (trainx3d.shape[1], 1)
    
    logger.info(f"Data Prepared. 2D Shape: {trainx2d.shape}, 3D Shape: {trainx3d.shape}")
    
    model_configure = [
        {
            'name': 'MLP',
            'builder': models.create_mlp_model,
            'fit_args': {'x': trainx2d, 'y': trainy},
            'val_args': (testx2d, testy),
            'input_shape': input_shape_2d
        },
        {
            'name': 'CNN_1D',
            'builder': models.create_cnn1d_model,
            'fit_args': {'x': trainx3d, 'y': trainy},
            'val_args': (testx3d, testy),
            'input_shape': input_shape_3d
        },
        {
            'name': 'LSTM',
            'builder': models.create_lstm_model,
            'fit_args': {'x': trainx3d, 'y': trainy},
            'val_args': (testx3d, testy),
            'input_shape': input_shape_3d
        }
    ]
    
    results = []
    log_file_path = os.path.join(cfg.RESULTS_PATH, 'experiment_log.txt')
    
    with open(log_file_path, 'w') as log_f:
        log_f.write("model configuration\n")
        for c in model_configure:
            name = c['name']
            logger.info(f"Training Start:{name}")
            logger.info(f"HyperParameters: {c}")
            log_f.write(f"Model:{name}\n")

            model = c['builder'](input_shape=c['input_shape'], num_classes=num)
            
            # callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
            ]
            
            history = model.fit(
                c['fit_args']['x'], c['fit_args']['y'],
                validation_data=c['val_args'],
                epochs=60, 
                batch_size=32, 
                callbacks=callbacks, 
                verbose=1
            )
            
            logger.info(f"Training finished for {name}. Max Val Acc: {max(history.history['val_accuracy']):.4f}")
            model_path = os.path.join(cfg.MODELS_PATH, f'{name}_model.h5')
            model.save(model_path)
            log_f.write(f"Saved to: {model_path}\n")

            xval, yval = c['val_args']
            loss, acc = model.evaluate(xval, yval, verbose=0)
            ypred_probab = model.predict(xval)
            ypred = np.argmax(ypred_probab, axis=1)
            
            report = classification_report(testy, ypred, target_names=le.classes_)
            precision = precision_score(testy, ypred, average='weighted')
            recall = recall_score(testy, ypred, average='weighted')
            f1 = f1_score(testy, ypred, average='weighted')

            log_f.write(f"  Test Accuracy: {acc:.4f}\n")
            log_f.write(f"  Precision: {precision:.4f}\n")
            log_f.write(f"  Recall: {recall:.4f}\n")
            log_f.write(f"  F1 Score: {f1:.4f}\n")
            log_f.write("  Classification Report:\n")
            log_f.write(report + "\n\n")
            logger.info(f"Evaluation Metrics for {name}:\nAccuracy: {acc:.4f}\n{report}")
            results.append({'Model': name, 'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1': f1})

            v.plot_training_history(history, name)
            v.plot_confusion_matrix(testy, ypred, le.classes_, name)
            v.plot_roc_curves(testy, ypred_probab, le.classes_, name)
            v.plot_tsne(model, xval, testy, le.classes_, name)

    logger.info("Generating Comparision for Research Justification")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(cfg.RESULTS_PATH, 'model_comparison_metrics.csv'), index=False)
    v.plot_model_comparison(results_df)

    with open(os.path.join(cfg.RESULTS_PATH, 'research_justification.txt'), 'w') as f:
        f.write("# Research Justification\n")
        f.write(results_df.to_markdown())
    
    logger.info("Check 'results/' for logs, plots, and justification.")

if __name__=="__main__":
    evaluate()
