import os 
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
import config as cfg
import data_processing as dp 
import models 
import visualizations as v
import logging 

logger = logging.getLogger(__name__)

np.random.seed(cfg.RANDOM_SEED)
tf.random.set_seed(cfg.RANDOM_SEED)

def prepare_data():
    logger.info("Preparing data")
    x, y = dp.load_data()
    if x.shape[0] == 0:
        raise ValueError("No data found")
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED, stratify=y
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=cfg.VALIDATION_SPLIT / (1 - cfg.TEST_SIZE), 
        random_state=cfg.RANDOM_SEED, stratify=y_temp
    )
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_val = sc.transform(x_val)
    x_test = sc.transform(x_test)
    
    joblib.dump(sc, os.path.join(cfg.MODELS_PATH, 'scaler.pkl'))
    joblib.dump(le, os.path.join(cfg.MODELS_PATH, 'label_encoder.pkl'))
    
    logger.info(f"Data split. Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    logger.info(f"Classes encoded: {list(zip(range(len(le.classes_)), le.classes_))}")

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"Class weights: {class_weight_dict}")

    return x_train, x_val, x_test, y_train, y_val, y_test, le, class_weight_dict

def reshape1(x):
    return x.reshape(x.shape[0], x.shape[1], 1)

def evaluate():
    x_train_2d, x_val_2d, x_test_2d, y_train, y_val, y_test, le, class_weights = prepare_data()
    num = len(np.unique(y_train))
    input_shape_2d = (x_train_2d.shape[1],)
    
    x_train_3d = reshape1(x_train_2d)
    x_val_3d = reshape1(x_val_2d)
    x_test_3d = reshape1(x_test_2d)
    input_shape_3d = (x_train_3d.shape[1], 1)
    
    logger.info(f"Data Prepared. 2D Shape: {x_train_2d.shape}, 3D Shape: {x_train_3d.shape}")
    
    model_configure = [
        {
            'name': 'MLP',
            'builder': models.create_mlp_model,
            'train_data': (x_train_2d, y_train),
            'val_data': (x_val_2d, y_val),
            'test_data': (x_test_2d, y_test),
            'input_shape': input_shape_2d,
            'params': {'num_units': 512, 'dropout_rate': 0.4, 'num_layers': 3, 'learning_rate': 0.0005}
        },
        {
            'name': 'CNN_1D',
            'builder': models.create_cnn1d_model,
            'train_data': (x_train_3d, y_train),
            'val_data': (x_val_3d, y_val),
            'test_data': (x_test_3d, y_test),
            'input_shape': input_shape_3d,
            'params': {'filters': 128, 'kernel_size': 5, 'dropout_rate': 0.4, 'learning_rate': 0.0005}
        },
        {
            'name': 'LSTM',
            'builder': models.create_lstm_model,
            'train_data': (x_train_3d, y_train),
            'val_data': (x_val_3d, y_val),
            'test_data': (x_test_3d, y_test),
            'input_shape': input_shape_3d,
            'params': {'units': 256, 'dropout_rate': 0.4, 'learning_rate': 0.0005}
        },
        {
            'name': 'Hybrid_CNN_LSTM',
            'builder': models.create_hybrid_model,
            'train_data': (x_train_3d, y_train),
            'val_data': (x_val_3d, y_val),
            'test_data': (x_test_3d, y_test),
            'input_shape': input_shape_3d,
            'params': {'filters': 128, 'lstm_units': 128, 'dropout_rate': 0.5, 'learning_rate': 0.0003}
        }
    ]
    
    results = []
    all_histories = {}
    log_file_path = os.path.join(cfg.RESULTS_PATH, 'experiment_log.txt')
    
    with open(log_file_path, 'w') as log_f:
        log_f.write("Model Training Results\n")
        log_f.write("=" * 80 + "\n\n")
        
        for c in model_configure:
            name = c['name']
            logger.info(f"Training Start: {name}")
            logger.info(f"HyperParameters: {c['params']}")
            log_f.write(f"Model: {name}\n")
            log_f.write(f"Parameters: {c['params']}\n")

            model = c['builder'](input_shape=c['input_shape'], num_classes=num, **c['params'])
            
            checkpoint_path = os.path.join(cfg.MODELS_PATH, f'{name}_best.h5')
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", 
                    patience=15, 
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=7, 
                    min_lr=1e-6,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
            ]
            
            x_train, y_train_data = c['train_data']
            x_val, y_val_data = c['val_data']
            
            history = model.fit(
                x_train, y_train_data,
                validation_data=(x_val, y_val_data),
                epochs=cfg.EPOCHS, 
                batch_size=cfg.BATCH_SIZE, 
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            all_histories[name] = history.history
            
            logger.info(f"Training finished for {name}. Max Val Acc: {max(history.history['val_accuracy']):.4f}")
            model_path = os.path.join(cfg.MODELS_PATH, f'{name}_model.h5')
            model.save(model_path)
            log_f.write(f"Saved to: {model_path}\n")

            x_test, y_test_data = c['test_data']
            loss, acc = model.evaluate(x_test, y_test_data, verbose=0)
            y_pred_probab = model.predict(x_test)
            y_pred = np.argmax(y_pred_probab, axis=1)
            
            report = classification_report(y_test_data, y_pred, target_names=le.classes_)
            precision = precision_score(y_test_data, y_pred, average='weighted')
            recall = recall_score(y_test_data, y_pred, average='weighted')
            f1 = f1_score(y_test_data, y_pred, average='weighted')

            log_f.write(f"  Test Accuracy: {acc:.4f}\n")
            log_f.write(f"  Precision: {precision:.4f}\n")
            log_f.write(f"  Recall: {recall:.4f}\n")
            log_f.write(f"  F1 Score: {f1:.4f}\n")
            log_f.write("  Classification Report:\n")
            log_f.write(report + "\n\n")
            logger.info(f"Evaluation Metrics for {name}:\nAccuracy: {acc:.4f}\n{report}")
            
            results.append({
                'Model': name, 
                'Accuracy': acc, 
                'Precision': precision, 
                'Recall': recall, 
                'F1': f1,
                'Val_Accuracy': max(history.history['val_accuracy']),
                'Train_Accuracy': max(history.history['accuracy'])
            })

            v.plot_training_history(history, name)
            v.plot_confusion_matrix(y_test_data, y_pred, le.classes_, name)
            v.plot_roc_curves(y_test_data, y_pred_probab, le.classes_, name)
            v.plot_tsne(model, x_test, y_test_data, le.classes_, name)
            
            log_f.write("-" * 80 + "\n\n")

    logger.info("Generating Comparison for Research Justification")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(cfg.RESULTS_PATH, 'model_comparison_metrics.csv'), index=False)
    
    v.plot_model_comparison(results_df)
    v.plot_all_training_comparison(all_histories)
    v.plot_metrics_radar(results_df)
    v.plot_class_performance(y_test, y_pred, le.classes_)

    with open(os.path.join(cfg.RESULTS_PATH, 'research_justification.txt'), 'w') as f:
        f.write("# Speech Emotion Recognition - Research Justification\n\n")
        f.write("## Model Performance Comparison\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n## Best Model\n")
        best_model = results_df.loc[results_df['F1'].idxmax()]
        f.write(f"Model: {best_model['Model']}\n")
        f.write(f"Accuracy: {best_model['Accuracy']:.4f}\n")
        f.write(f"F1 Score: {best_model['F1']:.4f}\n")
    
    logger.info("Check 'results/' for logs, plots, and justification.")

if __name__=="__main__":
    evaluate()
